#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types
from funasr.build_utils.build_model_from_file import build_model_from_file
from funasr.models.frontend.wav_frontend import WavFrontend
from funasr.text.token_id_converter import TokenIDConverter
from funasr.torch_utils.device_funcs import to_device


class Speech2Timestamp:
    def __init__(
            self,
            timestamp_infer_config: Union[Path, str] = None,
            timestamp_model_file: Union[Path, str] = None,
            timestamp_cmvn_file: Union[Path, str] = None,
            device: str = "cpu",
            dtype: str = "float32",
            **kwargs,
    ):
        assert check_argument_types()
        # 1. Build ASR model
        tp_model, tp_train_args = build_model_from_file(
            timestamp_infer_config, timestamp_model_file, cmvn_file=None, device=device, task_name="asr", mode="tp"
        )
        if 'cuda' in device:
            tp_model = tp_model.cuda()  # force model to cuda

        frontend = None
        if tp_train_args.frontend is not None:
            frontend = WavFrontend(cmvn_file=timestamp_cmvn_file, **tp_train_args.frontend_conf)

        logging.info("tp_model: {}".format(tp_model))
        logging.info("tp_train_args: {}".format(tp_train_args))
        tp_model.to(dtype=getattr(torch, dtype)).eval()

        logging.info(f"Decoding device={device}, dtype={dtype}")

        self.tp_model = tp_model
        self.tp_train_args = tp_train_args

        token_list = self.tp_model.token_list
        self.converter = TokenIDConverter(token_list=token_list)

        self.device = device
        self.dtype = dtype
        self.frontend = frontend
        self.encoder_downsampling_factor = 1
        if tp_train_args.encoder_conf["input_layer"] == "conv2d":
            self.encoder_downsampling_factor = 4

    @torch.no_grad()
    def __call__(
            self,
            speech: Union[torch.Tensor, np.ndarray],
            speech_lengths: Union[torch.Tensor, np.ndarray] = None,
            text_lengths: Union[torch.Tensor, np.ndarray] = None
    ):
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        if self.frontend is not None:
            feats, feats_len = self.frontend.forward(speech, speech_lengths)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
            self.tp_model.frontend = None
        else:
            feats = speech
            feats_len = speech_lengths

        # lfr_factor = max(1, (feats.size()[-1]//80)-1)
        batch = {"speech": feats, "speech_lengths": feats_len}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, enc_len = self.tp_model.encode(**batch)
        if isinstance(enc, tuple):
            enc = enc[0]

        # c. Forward Predictor
        _, _, us_alphas, us_peaks = self.tp_model.calc_predictor_timestamp(enc, enc_len,
                                                                           text_lengths.to(self.device) + 1)
        return us_alphas, us_peaks
