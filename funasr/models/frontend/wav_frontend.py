# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from espnet/espnet.

import copy
from typing import Optional, Tuple, Union

import humanfriendly
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.layers.log_mel import LogMel
from funasr.layers.stft import Stft
from funasr.utils.get_default_kwargs import get_default_kwargs
from funasr.modules.frontends.frontend import Frontend
from typeguard import check_argument_types


def apply_cmvn(inputs, mvn):  # noqa
    """
    Apply CMVN with mvn data
    """

    device = inputs.device
    dtype = inputs.dtype
    frame, dim = inputs.shape

    meams = np.tile(mvn[0:1, :dim], (frame, 1))
    vars = np.tile(mvn[1:2, :dim], (frame, 1))
    inputs += torch.from_numpy(meams).type(dtype).to(device)
    inputs *= torch.from_numpy(vars).type(dtype).to(device)

    return inputs.type(torch.float32)


def apply_lfr(inputs, lfr_m, lfr_n):
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    for i in range(T_lfr):
        if lfr_m <= T - i * lfr_n:
            LFR_inputs.append((inputs[i * lfr_n:i * lfr_n + lfr_m]).view(1, -1))
        else:  # process last LFR frame
            num_padding = lfr_m - (T - i * lfr_n)
            frame = (inputs[i * lfr_n:]).view(-1)
            for _ in range(num_padding):
                frame = torch.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    LFR_outputs = torch.vstack(LFR_inputs)
    return LFR_outputs.type(torch.float32)


class WavFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.
    """
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        window: Optional[str] = 'hamming',
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        lfr_m: int = 1,
        lfr_n: int = 1,
        htk: bool = False,
        mvn_data=None,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.fs = fs
        self.mvn_data = mvn_data
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.frontend_type = 'default'

    def output_size(self) -> int:
        return self.n_mels

    def forward(
            self, input: torch.Tensor,
            input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        sample_frequency = self.fs
        num_mel_bins = self.n_mels
        frame_length = self.win_length * 1000 / sample_frequency
        frame_shift = self.hop_length * 1000 / sample_frequency

        waveform = input * (1 << 15)

        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=1.0,
                          energy_floor=0.0,
                          window_type=self.window,
                          sample_frequency=sample_frequency)
        if self.lfr_m != 1 or self.lfr_n != 1:
            mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
        if self.mvn_data is not None:
            mat = apply_cmvn(mat, self.mvn_data)

        input_feats = mat[None, :]
        feats_lens = torch.randn(1)
        feats_lens.fill_(input_feats.shape[1])

        return input_feats, feats_lens
