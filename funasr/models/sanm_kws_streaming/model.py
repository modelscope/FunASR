#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import time
import torch
import logging
from typing import Dict, Tuple
from contextlib import contextmanager
from distutils.version import LooseVersion

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils import postprocess_utils
from funasr.metrics.compute_acc import th_accuracy
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.sanm_kws.model import SanmKWS
from funasr.models.paraformer.search import Hypothesis
from funasr.models.paraformer.cif_predictor import mae_loss
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


@tables.register("model_classes", "SanmKWSStreaming")
class SanmKWSStreaming(SanmKWS):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        decoding_ind = kwargs.get("decoding_ind")
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        # Encoder
        if hasattr(self.encoder, "overlap_chunk_cls"):
            ind = self.encoder.overlap_chunk_cls.random_choice(self.training, decoding_ind)
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, ind=ind)
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # decoder: CTC branch
        if hasattr(self.encoder, "overlap_chunk_cls"):
            encoder_out_ctc, encoder_out_lens_ctc = self.encoder.overlap_chunk_cls.remove_chunk(
                encoder_out, encoder_out_lens, chunk_outs=None
            )
        else:
            encoder_out_ctc, encoder_out_lens_ctc = encoder_out, encoder_out_lens

        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out_ctc, encoder_out_lens_ctc, text, text_lengths
        )

        # Collect CTC branch stats
        stats = dict()
        stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
        stats["cer_ctc"] = cer_ctc

        loss = loss_ctc

        stats["cer"] = cer_ctc
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode_chunk(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        cache: dict = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):
            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        # Forward encoder
        encoder_out, encoder_out_lens, _ = self.encoder.forward_chunk(
            speech, speech_lengths, cache=cache["encoder"]
        )

        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, torch.tensor([encoder_out.size(1)])

    def init_cache(self, cache: dict = {}, **kwargs):
        chunk_size = kwargs.get("chunk_size", [0, 10, 5])
        encoder_chunk_look_back = kwargs.get("encoder_chunk_look_back", 0)
        decoder_chunk_look_back = kwargs.get("decoder_chunk_look_back", 0)
        batch_size = 1

        enc_output_size = kwargs["encoder_conf"]["output_size"]
        feats_dims = kwargs["frontend_conf"]["n_mels"] * kwargs["frontend_conf"]["lfr_m"]
        cache_encoder = {
            "start_idx": 0,
            "cif_hidden": torch.zeros((batch_size, 1, enc_output_size)),
            "cif_alphas": torch.zeros((batch_size, 1)),
            "encoder_out": None,
            "encoder_out_lens": None,
            "chunk_size": chunk_size,
            "encoder_chunk_look_back": encoder_chunk_look_back,
            "last_chunk": False,
            "opt": None,
            "feats": torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims)),
            "tail_chunk": False,
        }
        cache["encoder"] = cache_encoder

        cache_decoder = {
            "decode_fsmn": None,
            "decoder_chunk_look_back": decoder_chunk_look_back,
            "opt": None,
            "chunk_size": chunk_size,
        }
        cache["decoder"] = cache_decoder
        cache["frontend"] = {}
        cache["prev_samples"] = torch.empty(0)

        return cache

    def generate_chunk(
        self,
        speech,
        speech_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        cache = kwargs.get("cache", {})
        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        # Encoder
        is_final = kwargs.get("is_final", False)
        encoder_out, encoder_out_lens = self.encode_chunk(
            speech, speech_lengths, cache=cache, is_final=is_final
        )
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        chunk_size = cache["encoder"]["chunk_size"]
        real_start_pos = chunk_size[0]

        if encoder_out_lens[0] > chunk_size[0] + chunk_size[1] + chunk_size[2]:
            assert False, print("impossible case 1 !")
        if encoder_out_lens[0] == chunk_size[0] + chunk_size[1] + chunk_size[2]:
            real_end_pos = chunk_size[0] + chunk_size[1]
        elif encoder_out_lens[0] > chunk_size[0] + chunk_size[1]:
            real_end_pos = chunk_size[0] + chunk_size[1]
        elif encoder_out_lens[0] > chunk_size[0]:
            real_end_pos = encoder_out_lens[0]
        else:
            assert False, print("impossible case 2 !")

        encoder_out_accum = cache["encoder"]["encoder_out"]
        if encoder_out_accum is not None:
            encoder_out_accum = torch.cat((encoder_out_accum, encoder_out[:, real_start_pos:real_end_pos, :]), dim=1)
        else:
            encoder_out_accum = encoder_out[:, real_start_pos:real_end_pos, :]
        cache["encoder"]["encoder_out"] = encoder_out_accum

        if cache["encoder"]["encoder_out_lens"] is not None:
            cache["encoder"]["encoder_out_lens"][0] += real_end_pos - real_start_pos
        else:
            cache["encoder"]["encoder_out_lens"] = encoder_out_lens
            cache["encoder"]["encoder_out_lens"][0] = real_end_pos - real_start_pos

        if is_final:
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))

            results = []
            for i in range(encoder_out_accum.size(0)):
                x = encoder_out_accum[i, : cache["encoder"]["encoder_out_lens"][i], :]
                detect_result = self.kws_decoder.decode(x)
                is_deted, det_keyword, det_score = detect_result[0], detect_result[1], detect_result[2]

                if is_deted:
                    self.writer["detect"][key[i]] = "detected " + det_keyword + " " + str(det_score)
                    det_info = "detected " + det_keyword + " " + str(det_score)
                else:
                    self.writer["detect"][key[i]] = "rejected"
                    det_info = "rejected"

                result_i = {"key": key[i], "text": det_info}
                results.append(result_i)

            return results
        else:
            return None

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        cache: dict = {},
        **kwargs,
    ):
        keywords = kwargs.get("keywords")
        from funasr.utils.kws_utils import KwsCtcPrefixDecoder
        self.kws_decoder = KwsCtcPrefixDecoder(
            ctc=self.ctc,
            keywords=keywords,
            token_list=tokenizer.token_list,
            seg_dict=tokenizer.seg_dict,
        )

        meta_data = {}
        chunk_size = kwargs["chunk_size"]
        chunk_stride_samples = int(chunk_size[1] * 960)  # 600ms
        first_chunk_padding_samples = int(chunk_size[2] * 960)  # 600ms

        if len(cache) == 0:
            self.init_cache(cache, **kwargs)

        time1 = time.perf_counter()
        cfg = {"is_final": kwargs.get("is_final", False)}
        audio_sample_list = load_audio_text_image_video(
            data_in,
            fs=frontend.fs,
            audio_fs=kwargs.get("fs", 16000),
            data_type=kwargs.get("data_type", "sound"),
            tokenizer=tokenizer,
            cache=cfg,
        )
        _is_final = cfg["is_final"]  # if data_in is a file or url, set is_final=True

        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        assert len(audio_sample_list) == 1, "batch_size must be set 1"

        audio_sample = torch.cat((cache["prev_samples"], audio_sample_list[0]))

        if len(audio_sample) < first_chunk_padding_samples:
            print("key: {}, audio is too short for inference {}".format(key, len(audio_sample)))

        audio_sample_pre = audio_sample[0 : first_chunk_padding_samples]
        feat_pre, feat_pre_lengths = extract_fbank(
            [audio_sample_pre],
            data_type=kwargs.get("data_type", "sound"),
            frontend=frontend,
            cache=cache["frontend"],
            is_final=False,
        )

        audio_sample = audio_sample[first_chunk_padding_samples:]
        audio_chunks = int(len(audio_sample) // chunk_stride_samples)

        for i in range(audio_chunks):
            if i == 0:
                cache["encoder"]["feats"][:, chunk_size[2] :, :] = feat_pre

            kwargs["is_final"] = False
            audio_sample_i = audio_sample[i * chunk_stride_samples : (i + 1) * chunk_stride_samples]

            if kwargs["is_final"] and len(audio_sample_i) < 960:
                cache["encoder"]["tail_chunk"] = True
                speech = cache["encoder"]["feats"]
                speech_lengths = torch.tensor([speech.shape[1]], dtype=torch.int64).to(
                    speech.device
                )
            else:
                # extract fbank feats
                speech, speech_lengths = extract_fbank(
                    [audio_sample_i],
                    data_type=kwargs.get("data_type", "sound"),
                    frontend=frontend,
                    cache=cache["frontend"],
                    is_final=kwargs["is_final"],
                )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

            results_chunk_i = self.generate_chunk(
                speech,
                speech_lengths,
                key=key,
                tokenizer=tokenizer,
                cache=cache,
                frontend=frontend,
                **kwargs,
            )

            # results_chunk_i must be None when is_final=False
            assert results_chunk_i is None

        # process tail samples
        tail_audio_sample = audio_sample[ audio_chunks * chunk_stride_samples: ]
        if len(tail_audio_sample) < 960:
            kwargs["is_final"] = _is_final
            cache["encoder"]["tail_chunk"] = True
            speech = cache["encoder"]["feats"]
            speech_lengths = torch.tensor([speech.shape[1]], dtype=torch.int64).to(
                speech.device
            )
            results_chunk_tail = self.generate_chunk(
                speech,
                speech_lengths,
                key=key,
                tokenizer=tokenizer,
                cache=cache,
                frontend=frontend,
                **kwargs,
            )
        elif len(tail_audio_sample) <= first_chunk_padding_samples:
            kwargs["is_final"] = _is_final
            # extract fbank feats
            # cache["encoder"]["tail_chunk"] = True  # cannot be true
            speech, speech_lengths = extract_fbank(
                [ tail_audio_sample ],
                data_type=kwargs.get("data_type", "sound"),
                frontend=frontend,
                cache=cache["frontend"],
                is_final=kwargs["is_final"],
            )
            results_chunk_tail = self.generate_chunk(
                speech,
                speech_lengths,
                key=key,
                tokenizer=tokenizer,
                cache=cache,
                frontend=frontend,
                **kwargs,
            )
        elif len(tail_audio_sample) > first_chunk_padding_samples and \
             len(tail_audio_sample) < chunk_stride_samples:
            kwargs["is_final"] = False
            # extract fbank feats
            speech, speech_lengths = extract_fbank(
                [ tail_audio_sample ],
                data_type=kwargs.get("data_type", "sound"),
                frontend=frontend,
                cache=cache["frontend"],
                is_final=kwargs["is_final"],
            )
            results_chunk = self.generate_chunk(
                speech,
                speech_lengths,
                key=key,
                tokenizer=tokenizer,
                cache=cache,
                frontend=frontend,
                **kwargs,
            )
            # results_chunk must be None when is_final=False
            assert results_chunk is None

            # push tail chunk
            kwargs["is_final"] = _is_final
            cache["encoder"]["tail_chunk"] = True
            speech = cache["encoder"]["feats"]
            speech_lengths = torch.tensor([speech.shape[1]], dtype=torch.int64).to(
                speech.device
            )
            results_chunk_tail = self.generate_chunk(
                speech,
                speech_lengths,
                key=key,
                tokenizer=tokenizer,
                cache=cache,
                frontend=frontend,
                **kwargs,
            )

        result = results_chunk_tail

        if _is_final:
            self.init_cache(cache, **kwargs)

        if kwargs.get("output_dir"):
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))

        return result, meta_data

    def export(self, **kwargs):
        from .export_meta import export_rebuild_model

        models = export_rebuild_model(model=self, **kwargs)
        return models
