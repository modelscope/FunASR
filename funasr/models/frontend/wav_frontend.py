# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from espnet/espnet.

from typing import Tuple

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from funasr.models.frontend.abs_frontend import AbsFrontend
from typeguard import check_argument_types
from torch.nn.utils.rnn import pad_sequence


def load_cmvn(cmvn_file):
    with open(cmvn_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == '<AddShift>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                add_shift_line = line_item[3:(len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == '<Rescale>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                rescale_line = line_item[3:(len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float)
    vars = np.array(vars_list).astype(np.float)
    cmvn = np.array([means, vars])
    cmvn = torch.as_tensor(cmvn) 
    return cmvn 
          

def apply_cmvn(inputs, cmvn_file):  # noqa
    """
    Apply CMVN with mvn data
    """

    device = inputs.device
    dtype = inputs.dtype
    frame, dim = inputs.shape

    cmvn = load_cmvn(cmvn_file)
    means = np.tile(cmvn[0:1, :dim], (frame, 1))
    vars = np.tile(cmvn[1:2, :dim], (frame, 1))
    inputs += torch.from_numpy(means).type(dtype).to(device)
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
        cmvn_file: str = None,
        fs: int = 16000,
        window: str = 'hamming',
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        filter_length_min: int = -1,
        filter_length_max: int = -1,
        lfr_m: int = 1,
        lfr_n: int = 1,
        dither: float = 1.0,
        snip_edges: bool = True,
        upsacle_samples: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        self.fs = fs
        self.window = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.filter_length_min = filter_length_min
        self.filter_length_max = filter_length_max
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file
        self.dither = dither
        self.snip_edges = snip_edges
        self.upsacle_samples = upsacle_samples

    def output_size(self) -> int:
        return self.n_mels * self.lfr_m

    def forward(
            self,
            input: torch.Tensor,
            input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            if self.upsacle_samples:
                waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(waveform,
                              num_mel_bins=self.n_mels,
                              frame_length=self.frame_length,
                              frame_shift=self.frame_shift,
                              dither=self.dither,
                              energy_floor=0.0,
                              window_type=self.window,
                              sample_frequency=self.fs,
                              snip_edges=self.snip_edges)
     
            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            if self.cmvn_file is not None:
                mat = apply_cmvn(mat, self.cmvn_file) 
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats,
                                 batch_first=True,
                                 padding_value=0.0)
        return feats_pad, feats_lens

    def forward_fbank(
            self,
            input: torch.Tensor,
            input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(waveform,
                              num_mel_bins=self.n_mels,
                              frame_length=self.frame_length,
                              frame_shift=self.frame_shift,
                              dither=self.dither,
                              energy_floor=0.0,
                              window_type=self.window,
                              sample_frequency=self.fs)


            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats,
                                 batch_first=True,
                                 padding_value=0.0)
        return feats_pad, feats_lens

    def forward_lfr_cmvn(
            self,
            input: torch.Tensor,
            input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            mat = input[i, :input_lengths[i], :]
            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            if self.cmvn_file is not None:
                mat = apply_cmvn(mat, self.cmvn_file)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats,
                                 batch_first=True,
                                 padding_value=0.0)
        return feats_pad, feats_lens
