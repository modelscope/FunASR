#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import time
import torch
import logging
from torch.cuda.amp import autocast
from typing import Union, Dict, List, Tuple, Optional

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils import postprocess_utils
from funasr.metrics.compute_acc import th_accuracy
from funasr.train_utils.device_funcs import to_device
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.search import Hypothesis
from funasr.models.paraformer.cif_predictor import mae_loss
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank


@tables.register("model_classes", "SanmKWS")
class SanmKWS(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        specaug: Optional[str] = None,
        specaug_conf: Optional[Dict] = None,
        normalize: str = None,
        normalize_conf: Optional[Dict] = None,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        ctc: str = None,
        ctc_conf: Optional[Dict] = None,
        ctc_weight: float = 1.0,
        input_size: int = 360,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        **kwargs,
    ):

        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)

        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)

        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if ctc_conf is None:
            ctc_conf = {}
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)

        # note that eos is the same as sos (equivalent ID)
        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        # self.token_list = token_list.copy()
        #
        # self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder

        self.ctc = ctc
        self.error_calculator = None

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
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]
        batch_size = speech.shape[0]

        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # decoder: CTC branch
        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )

        # Collect CTC branch stats
        stats = dict()
        stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
        stats["cer_ctc"] = cer_ctc
        stats["cer"] = cer_ctc

        loss = loss_ctc

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder. Note that this method is used by asr_inference.py
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
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
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
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is not None:
                speech_lengths = speech_lengths.squeeze(-1)
            else:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])
        # Encoder
        if kwargs.get("fp16", False):
            speech = speech.half()
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        results = []
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))

        for i in range(encoder_out.size(0)):
            x = encoder_out[i, : encoder_out_lens[i], :]
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

        return results, meta_data

    def export(self, **kwargs):
        from .export_meta import export_rebuild_model

        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        models = export_rebuild_model(model=self, **kwargs)
        return models
