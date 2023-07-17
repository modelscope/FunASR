#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import logging
import math
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
import wave

import numpy as np
import torch

from funasr.build_utils.build_model_from_file import build_model_from_file
from funasr.models.frontend.wav_frontend import WavFrontend, WavFrontendOnline
from funasr.torch_utils.device_funcs import to_device
from funasr.runtime.python.onnxruntime.funasr_onnx.utils.e2e_vad import E2EVadModel


class Speech2VadSegment:
    """Speech2VadSegment class

    Examples:
        >>> import soundfile
        >>> speech2segment = Speech2VadSegment("vad_config.yml", "vad.pt")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2segment(audio)
        [[10, 230], [245, 450], ...]

    """

    def __init__(
            self,
            vad_infer_config: Union[Path, str] = None,
            vad_model_file: Union[Path, str] = None,
            vad_cmvn_file: Union[Path, str] = None,
            device: str = "cpu",
            batch_size: int = 1,
            dtype: str = "float32",
            **kwargs,
    ):

        # 1. Build vad model
        vad_model, vad_infer_args = build_model_from_file(
            vad_infer_config, vad_model_file, None, device, task_name="vad"
        )
        frontend = None
        if vad_infer_args.frontend is not None:
            frontend = WavFrontend(cmvn_file=vad_cmvn_file, **vad_infer_args.frontend_conf)

        logging.info("vad_model: {}".format(vad_model))
        logging.info("vad_infer_args: {}".format(vad_infer_args))
        vad_model.to(dtype=getattr(torch, dtype)).eval()

        self.vad_model = vad_model
        self.vad_infer_args = vad_infer_args
        self.device = device
        self.dtype = dtype
        self.frontend = frontend
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None,
            in_cache: Dict[str, torch.Tensor] = dict()
    ) -> Tuple[List[List[int]], Dict[str, torch.Tensor]]:
        """Inference

        Args:
            speech: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            self.frontend.filter_length_max = math.inf
            fbanks, fbanks_len = self.frontend.forward_fbank(speech, speech_lengths)
            feats, feats_len = self.frontend.forward_lfr_cmvn(fbanks, fbanks_len)
            fbanks = to_device(fbanks, device=self.device)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
        else:
            raise Exception("Need to extract feats first, please configure frontend configuration")

        # b. Forward Encoder streaming
        t_offset = 0
        step = min(feats_len.max(), 6000)
        segments = [[]] * self.batch_size
        for t_offset in range(0, feats_len, min(step, feats_len - t_offset)):
            if t_offset + step >= feats_len - 1:
                step = feats_len - t_offset
                is_final = True
            else:
                is_final = False
            batch = {
                "feats": feats[:, t_offset:t_offset + step, :],
                "waveform": speech[:, t_offset * 160:min(speech.shape[-1], (t_offset + step - 1) * 160 + 400)],
                "is_final": is_final,
                "in_cache": in_cache
            }
            # a. To device
            # batch = to_device(batch, device=self.device)
            segments_part, in_cache = self.vad_model(**batch)
            if segments_part:
                for batch_num in range(0, self.batch_size):
                    segments[batch_num] += segments_part[batch_num]
        return fbanks, segments


class Speech2VadSegment4ClipVideo:
    """Speech2VadSegment class

    Examples:
        >>> import soundfile
        >>> speech2segment = Speech2VadSegment("vad_config.yml", "vad.pt")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2segment(audio)
        [[10, 230], [245, 450], ...]

    """

    def __init__(
            self,
            vad_infer_config: Union[Path, str] = None,
            vad_model_file: Union[Path, str] = None,
            vad_cmvn_file: Union[Path, str] = None,
            device: str = "cpu",
            batch_size: int = 1,
            dtype: str = "float32",
            **kwargs,
    ):

        # 1. Build vad model
        vad_model, vad_infer_args = build_model_from_file(
            vad_infer_config, vad_model_file, None, device, task_name="vad"
        )
        frontend = None
        if vad_infer_args.frontend is not None:
            frontend = WavFrontend(cmvn_file=vad_cmvn_file, **vad_infer_args.frontend_conf)

        logging.info("vad_model: {}".format(vad_model))
        logging.info("vad_infer_args: {}".format(vad_infer_args))
        vad_model.to(dtype=getattr(torch, dtype)).eval()

        self.vad_model = vad_model
        self.vad_infer_args = vad_infer_args
        self.device = device
        self.dtype = dtype
        self.frontend = frontend
        self.batch_size = batch_size
        self.max_end_sil = vad_infer_args.vad_post_conf["max_end_silence_time"]

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None,
            in_cache: Dict[str, torch.Tensor] = dict()
    ) -> Tuple[List[List[int]], Dict[str, torch.Tensor]]:
        """Inference

        Args:
            speech: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        vad_scorer = E2EVadModel(self.vad_infer_args.vad_post_conf)
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if self.frontend is not None:
            self.frontend.filter_length_max = math.inf
            fbanks, fbanks_len = self.frontend.forward_fbank(speech, speech_lengths)
            feats, feats_len = self.frontend.forward_lfr_cmvn(fbanks, fbanks_len)
            fbanks = to_device(fbanks, device=self.device)
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
        else:
            raise Exception("Need to extract feats first, please configure frontend configuration")

        segments = [[]]
        # vad fsmn encoder forward in batch
        # split batch
        import time
        import copy
        start_time = time.time()
        raw_feats = copy.copy(feats)
        for i in range(20):
            feats = copy.copy(raw_feats)
            force_split = 1200
            _in_cache = {}
            T = feats.shape[1]
            D = feats.shape[-1]
            if force_split != 0 and T > force_split:
                # padding
                assert force_split > 100, "FSMN VAD has left context 80 frames, force_split>100 supposed."
                if T % force_split:
                    _to_pad = force_split - T % force_split
                    feats = torch.concat([feats, torch.zeros([1, _to_pad, D])], dim=1)
                else:
                    _to_pad = 0
                # reshape
                B = (T + _to_pad) // force_split
                feats = feats.reshape([B, force_split, D])
                # padding left and right context
                empty_left_context = torch.zeros(1, 80, D)
                left_context = torch.concat([empty_left_context, feats[:,-80:,:][:-1]], dim=0)
                right_context = feats[:,-1,:].unsqueeze(1)
                feats = torch.concat([left_context, feats, right_context, right_context], dim=1)
                scores = self.vad_model.encoder(feats, _in_cache).to('cpu')  # return B * T * D
                # recover frames
                new_D = scores.shape[-1]
                scores = scores[:, 80:force_split+80,].reshape(1, -1, new_D)[:,:T,:]
            else:
                scores = self.vad_model.encoder(feats, _in_cache).to('cpu')  # return B * T * D
            waveform = speech
            segments_part = vad_scorer(scores, waveform, is_final=True, max_end_sil=self.max_end_sil, online=False)
            if segments_part:
                for batch_num in range(0, self.batch_size):
                    segments[batch_num] += segments_part[batch_num]
        end_time = time.time()
        print(end_time-start_time)
        import pdb; pdb.set_trace()
        return fbanks, segments


class Speech2VadSegmentOnline(Speech2VadSegment):
    """Speech2VadSegmentOnline class

    Examples:
        >>> import soundfile
        >>> speech2segment = Speech2VadSegmentOnline("vad_config.yml", "vad.pt")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2segment(audio)
        [[10, 230], [245, 450], ...]

    """

    def __init__(self, **kwargs):
        super(Speech2VadSegmentOnline, self).__init__(**kwargs)
        vad_cmvn_file = kwargs.get('vad_cmvn_file', None)
        self.frontend = None
        if self.vad_infer_args.frontend is not None:
            self.frontend = WavFrontendOnline(cmvn_file=vad_cmvn_file, **self.vad_infer_args.frontend_conf)

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None,
            in_cache: Dict[str, torch.Tensor] = dict(), is_final: bool = False, max_end_sil: int = 800
    ) -> Tuple[torch.Tensor, List[List[int]], torch.Tensor]:
        """Inference

        Args:
            speech: Input speech data
        Returns:
            text, token, token_int, hyp

        """

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        batch_size = speech.shape[0]
        segments = [[]] * batch_size
        if self.frontend is not None:
            reset = in_cache == dict()
            feats, feats_len = self.frontend.forward(speech, speech_lengths, is_final, reset)
            fbanks, _ = self.frontend.get_fbank()
        else:
            raise Exception("Need to extract feats first, please configure frontend configuration")
        if feats.shape[0]:
            feats = to_device(feats, device=self.device)
            feats_len = feats_len.int()
            waveforms = self.frontend.get_waveforms()

            batch = {
                "feats": feats,
                "waveform": waveforms,
                "in_cache": in_cache,
                "is_final": is_final,
                "max_end_sil": max_end_sil
            }
            # a. To device
            batch = to_device(batch, device=self.device)
            segments, in_cache = self.vad_model.forward_online(**batch)
            # in_cache.update(batch['in_cache'])
            # in_cache = {key: value for key, value in batch['in_cache'].items()}
        return fbanks, segments, in_cache
