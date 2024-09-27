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
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.search import Hypothesis
from funasr.models.paraformer.cif_predictor import mae_loss
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank


@tables.register("model_classes", "FsmnKWSMT")
class FsmnKWSMT(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(
        self,
        specaug: Optional[str] = None,
        specaug_conf: Optional[Dict] = None,
        normalize: str = None,
        normalize_conf: Optional[Dict] = None,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        ctc_conf: Optional[Dict] = None,
        input_size: int = 360,
        vocab_size: list = [],
        ignore_id: int = -1,
        blank_id: int = 0,
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
        encoder = encoder_class(**encoder_conf)
        encoder_output_size = encoder.output_size()
        encoder_output_size2 = encoder.output_size2()

        ctc = CTC(
            odim=vocab_size[0], encoder_output_size=encoder_output_size, **ctc_conf
        )
        ctc2 = CTC(
            odim=vocab_size[1], encoder_output_size=encoder_output_size2, **ctc_conf
        )

        self.blank_id = blank_id
        self.ignore_id = ignore_id

        # self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.ctc = ctc
        self.ctc2 = ctc2

        self.error_calculator = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text2: torch.Tensor,
        text2_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
                text2: (Batch, Length)
                text2_lengths: (Batch,)
        """
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]
        batch_size = speech.shape[0]

        # Encoder
        encoder_out, encoder_out2, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )
        loss_ctc2, cer_ctc2 = self._calc_ctc_loss(
            encoder_out2, encoder_out_lens, text2, text2_lengths
        )

        # Collect CTC branch stats
        stats = dict()
        stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
        stats["cer_ctc"] = cer_ctc
        stats["loss_ctc2"] = loss_ctc2.detach() if loss_ctc2 is not None else None
        stats["cer_ctc2"] = cer_ctc2

        loss = 0.5 * loss_ctc + 0.5 * loss_ctc2

        stats["cer"] = cer_ctc
        stats["cer2"] = cer_ctc2
        stats["loss"] = torch.clone(loss.detach())

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
        encoder_out, encoder_out2 = self.encoder(speech)
        encoder_out_lens = speech_lengths

        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        if isinstance(encoder_out2, tuple):
            encoder_out2 = encoder_out2[0]

        return encoder_out, encoder_out2, encoder_out_lens

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

    def _calc_ctc2_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc2(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc2.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc


    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list=None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        keywords = kwargs.get("keywords")
        from funasr.utils.kws_utils import KwsCtcPrefixDecoder
        self.kws_decoder = KwsCtcPrefixDecoder(
            ctc=self.ctc,
            keywords=keywords,
            token_list=tokenizer[0].token_list,
            seg_dict=tokenizer[0].seg_dict,
        )
        self.kws_decoder2 = KwsCtcPrefixDecoder(
            ctc=self.ctc2,
            keywords=keywords,
            token_list=tokenizer[1].token_list,
            seg_dict=tokenizer[1].seg_dict,
        )

        meta_data = {}
        if isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank": # fbank
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
                tokenizer=tokenizer
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list,
                data_type=kwargs.get("data_type", "sound"),
                frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        # Encoder
        encoder_out, encoder_out2, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        if isinstance(encoder_out2, tuple):
            encoder_out2 = encoder_out2[0]

        results = []
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))

        for i in range(encoder_out.size(0)):
            x = encoder_out[i, :encoder_out_lens[i], :]
            detect_result = self.kws_decoder.decode(x)
            is_deted, det_keyword, det_score = detect_result[0], detect_result[1], detect_result[2]

            if is_deted:
                self.writer["detect"][key[i]] = "detected " + det_keyword + " " + str(det_score)
                det_info = "detected " + det_keyword + " " + str(det_score)
            else:
                self.writer["detect"][key[i]] = "rejected"
                det_info = "rejected"

            x2 = encoder_out2[i, :encoder_out_lens[i], :]
            detect_result2 = self.kws_decoder2.decode(x2)
            is_deted2, det_keyword2, det_score2 = detect_result2[0], detect_result2[1], detect_result2[2]

            if is_deted2:
                self.writer["detect2"][key[i]] = "detected " + det_keyword2 + " " + str(det_score2)
                det_info2 = "detected " + det_keyword2 + " " + str(det_score2)
            else:
                self.writer["detect2"][key[i]] = "rejected"
                det_info2 = "rejected"

            result_i = {"key": key[i], "text": det_info, "text2": det_info2}
            results.append(result_i)

        return results, meta_data


@tables.register("model_classes", "FsmnKWSMTConvert")
class FsmnKWSMTConvert(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(
        self,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        ctc_conf: Optional[Dict] = None,
        ctc_weight: float = 1.0,
        input_size: int = 360,
        blank_id: int = 0,
        **kwargs,
    ):
        super().__init__()

        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(**encoder_conf)
        encoder_output_size = encoder.output_size()
        self.blank_id = blank_id
        self.encoder = encoder

        self.error_calculator = None

    def to_kaldi_net(self):
        return self.encoder.to_kaldi_net()

    def to_kaldi_net2(self):
        return self.encoder.to_kaldi_net2()

    def to_pytorch_net(self, kaldi_file):
        return self.encoder.to_pytorch_net(kaldi_file)
