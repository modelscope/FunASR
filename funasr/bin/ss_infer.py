#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


import logging
from pathlib import Path
from typing import List
from typing import Union

import numpy as np
import torch

from funasr.build_utils.build_model_from_file import build_model_from_file
from funasr.torch_utils.device_funcs import to_device


class SpeechSeparator:
    """SpeechSeparator class

    Examples:
        >>> import soundfile
        >>> speech_separator = MossFormer("ss_config.yml", "ss.pt")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> separated_wavs = speech_separator(audio)        

    """

    def __init__(
            self,
            ss_infer_config: Union[Path, str] = None,
            ss_model_file: Union[Path, str] = None,
            device: str = "cpu",
            batch_size: int = 1,
            dtype: str = "float32",
            **kwargs,
    ):

        # 1. Build ss model
        ss_model, ss_infer_args = build_model_from_file(
            ss_infer_config, ss_model_file, None, device, task_name="ss"
        )

        logging.info("ss_model: {}".format(ss_model))
        logging.info("ss_infer_args: {}".format(ss_infer_args))

        ss_model.to(dtype=getattr(torch, dtype)).eval()

        self.ss_model = ss_model
        self.ss_infer_args = ss_infer_args
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

    def decode(self, model, args, inputs, nsamples):
        decode_do_segment = False
        with torch.no_grad():       
            out = []
            window = args.sample_rate * args.decode_window  # decoding window length
            stride = int(window*0.75)  # decoding stride if segmentation is used
            b, t = inputs.shape
            if t > window * args.one_time_decode_length:
                decode_do_segment = True  # set segment decoding to true for very long sequence

            if t < window:
                inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window-t))], 1)
            elif t < window + stride:
                padding = window + stride - t
                inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)
            else:
                if (t - window) % stride != 0:
                    padding = t - (t-window)//stride * stride
                    inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)
            inputs = torch.from_numpy(np.float32(inputs))
            inputs = to_device(inputs, device=self.device)
            b, t = inputs.shape
            if decode_do_segment:
                outputs = np.zeros((args.num_spks, t))
                give_up_length = (window - stride)//2
                current_idx = 0
                while current_idx + window <= t:
                    tmp_input = inputs[:, current_idx:current_idx+window]
                    tmp_out_list = model(tmp_input,)
                    for spk in range(args.num_spks):
                        tmp_out_list[spk] = tmp_out_list[spk][0, :].cpu().numpy()
                        if current_idx == 0:
                            outputs[spk, current_idx:current_idx+window-give_up_length] = \
                                tmp_out_list[spk][:-give_up_length]
                        else:
                            outputs[spk, current_idx+give_up_length:current_idx+window-give_up_length] = \
                                tmp_out_list[spk][give_up_length:-give_up_length]
                    current_idx += stride
                for spk in range(args.num_spks):
                    out.append(outputs[spk, :])
            else:
                out_list = model(inputs)
                for spk in range(args.num_spks):
                    out.append(out_list[spk][0, :].cpu().numpy())

            max_abs = 0
            for spk in range(args.num_spks):
                if max_abs < max(abs(out[spk])):
                    max_abs = max(abs(out[spk]))
            for spk in range(args.num_spks):
                out[spk] = out[spk][:nsamples]
                out[spk] = out[spk]/max_abs

        return out

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None,
    ) -> List[torch.Tensor]:
        """Inference

        Args:
            speech: Input speech data
        Returns:
            speech list: list of speech data

        """

        out = self.decode(self.ss_model, self.ss_infer_args, speech, speech_lengths)

        return out

