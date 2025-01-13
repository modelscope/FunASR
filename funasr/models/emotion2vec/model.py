#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
# Modified from https://github.com/ddlBoJack/emotion2vec/tree/main

import os
import time
import torch
import logging
import numpy as np
from functools import partial
from omegaconf import OmegaConf
import torch.nn.functional as F
from contextlib import contextmanager
from distutils.version import LooseVersion

from funasr.register import tables
from funasr.models.emotion2vec.modules import AltBlock
from funasr.models.emotion2vec.audio import AudioEncoder
from funasr.utils.load_utils import load_audio_text_image_video


logger = logging.getLogger(__name__)
if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


@tables.register("model_classes", "Emotion2vec")
class Emotion2vec(torch.nn.Module):
    """
    Author: Ziyang Ma, Zhisheng Zheng, Jiaxin Ye, Jinchao Li, Zhifu Gao, Shiliang Zhang, Xie Chen
    emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation
    https://arxiv.org/abs/2312.15185
    """

    def __init__(self, **kwargs):
        super().__init__()
        # import pdb; pdb.set_trace()
        cfg = OmegaConf.create(kwargs["model_conf"])
        self.cfg = cfg

        make_layer_norm = partial(
            torch.nn.LayerNorm, eps=cfg.get("norm_eps"), elementwise_affine=cfg.get("norm_affine")
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                cfg.get("embed_dim") if dim is None else dim,
                cfg.get("num_heads") if heads is None else heads,
                cfg.get("mlp_ratio"),
                qkv_bias=True,
                drop=cfg.get("encoder_dropout"),
                attn_drop=cfg.get("attention_dropout"),
                mlp_drop=cfg.get("activation_dropout"),
                post_mlp_drop=cfg.get("post_mlp_drop"),
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=cfg.get("layer_norm_first"),
                ffn_targets=not cfg.get("end_of_block_targets"),
            )

        self.alibi_biases = {}
        self.modality_encoders = torch.nn.ModuleDict()

        enc = AudioEncoder(
            cfg.modalities.audio,
            cfg.get("embed_dim"),
            make_block,
            make_layer_norm,
            cfg.get("layer_norm_first"),
            self.alibi_biases,
        )
        self.modality_encoders["AUDIO"] = enc

        self.ema = None

        self.average_top_k_layers = cfg.get("average_top_k_layers")
        self.loss_beta = cfg.get("loss_beta")
        self.loss_scale = cfg.get("loss_scale")

        self.dropout_input = torch.nn.Dropout(cfg.get("dropout_input"))

        dpr = np.linspace(
            cfg.get("start_drop_path_rate"), cfg.get("end_drop_path_rate"), cfg.get("depth")
        )

        self.blocks = torch.nn.ModuleList([make_block(dpr[i]) for i in range(cfg.get("depth"))])

        self.norm = None
        if cfg.get("layer_norm_first"):
            self.norm = make_layer_norm(cfg.get("embed_dim"))

        vocab_size = kwargs.get("vocab_size", -1)
        self.proj = None
        if vocab_size > 0:
            self.proj = torch.nn.Linear(cfg.get("embed_dim"), vocab_size)

    def forward(
        self,
        source,
        target=None,
        id=None,
        mode=None,
        padding_mask=None,
        mask=True,
        features_only=False,
        force_remove_masked=False,
        remove_extra_tokens=True,
        precomputed_mask=None,
        **kwargs,
    ):

        feature_extractor = self.modality_encoders["AUDIO"]

        mask_seeds = None

        extractor_out = feature_extractor(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.get("clone_batch") if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.cfg.get("layerdrop", 0) == 0
                or (np.random.random() > self.cfg.get("layerdrop", 0))
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = alibi_scale[i] if alibi_scale.size(0) > 1 else alibi_scale.squeeze(0)
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                if features_only:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        if features_only:
            if remove_extra_tokens:
                x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, feature_extractor.modality_cfg.num_extra_tokens :
                    ]

            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            }

    def extract_features(
        self, source, mode=None, padding_mask=None, mask=False, remove_extra_tokens=True
    ):
        res = self.forward(
            source,
            mode=mode,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            remove_extra_tokens=remove_extra_tokens,
        )
        return res

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        # if source_file.endswith('.wav'):
        #     wav, sr = sf.read(source_file)
        #     channel = sf.info(source_file).channels
        #     assert sr == 16e3, "Sample rate should be 16kHz, but got {}in file {}".format(sr, source_file)
        #     assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, source_file)
        granularity = kwargs.get("granularity", "utterance")
        extract_embedding = kwargs.get("extract_embedding", True)
        if self.proj is None:
            extract_embedding = True
        meta_data = {}
        # extract fbank feats
        time1 = time.perf_counter()
        audio_sample_list = load_audio_text_image_video(
            data_in,
            fs=16000,
            audio_fs=kwargs.get("fs", 16000),
            data_type=kwargs.get("data_type", "sound"),
            tokenizer=tokenizer,
        )
        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        meta_data["batch_data_time"] = len(audio_sample_list[0]) / kwargs.get("fs", 16000)

        results = []
        output_dir = kwargs.get("output_dir")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        for i, wav in enumerate(audio_sample_list):
            source = wav.to(device=kwargs["device"])
            if self.cfg.normalize:
                source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            feats = self.extract_features(source, padding_mask=None)
            x = feats["x"]
            feats = feats["x"].squeeze(0).cpu().numpy()
            if granularity == "frame":
                feats = feats
            elif granularity == "utterance":
                feats = np.mean(feats, axis=0)

            if output_dir and extract_embedding:
                np.save(os.path.join(output_dir, "{}.npy".format(key[i])), feats)

            labels = tokenizer.token_list if tokenizer is not None else []
            scores = []
            if self.proj:
                x = x.mean(dim=1)
                x = self.proj(x)
                for idx, lab in enumerate(labels):
                    x[:,idx] = -np.inf if lab.startswith("unuse") else x[:,idx]
                x = torch.softmax(x, dim=-1)
                scores = x[0].tolist()

            select_label = [lb for lb in labels if not lb.startswith("unuse")]
            select_score = [scores[idx] for idx, lb in enumerate(labels) if not lb.startswith("unuse")]

            # result_i = {"key": key[i], "labels": labels, "scores": scores}
            result_i = {"key": key[i], "labels": select_label, "scores": select_score}

            if extract_embedding:
                result_i["feats"] = feats
            results.append(result_i)

        return results, meta_data

    def export(self, **kwargs):
        from .export_meta import export_rebuild_model

        models = export_rebuild_model(model=self, **kwargs)
        return models
