# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from espnet/espnet.
from typing import Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence

import funasr.frontends.eend_ola_feature as eend_ola_feature
from funasr.register import tables


def load_cmvn(cmvn_file):
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                add_shift_line = line_item[3 : (len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                rescale_line = line_item[3 : (len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    cmvn = np.array([means, vars])
    cmvn = torch.as_tensor(cmvn, dtype=torch.float32)
    return cmvn


def apply_cmvn(inputs, cmvn):  # noqa
    """
    Apply CMVN with mvn data
    """

    device = inputs.device
    dtype = inputs.dtype
    frame, dim = inputs.shape

    means = cmvn[0:1, :dim]
    vars = cmvn[1:2, :dim]
    inputs += means.to(device)
    inputs *= vars.to(device)

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
            LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).view(1, -1))
        else:  # process last LFR frame
            num_padding = lfr_m - (T - i * lfr_n)
            frame = (inputs[i * lfr_n :]).view(-1)
            for _ in range(num_padding):
                frame = torch.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    LFR_outputs = torch.vstack(LFR_inputs)
    return LFR_outputs.type(torch.float32)


@tables.register("frontend_classes", "wav_frontend")
@tables.register("frontend_classes", "WavFrontend")
class WavFrontend(nn.Module):
    """Conventional frontend structure for ASR."""

    def __init__(
        self,
        cmvn_file: str = None,
        fs: int = 16000,
        window: str = "hamming",
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
        **kwargs,
    ):
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
        self.cmvn = None if self.cmvn_file is None else load_cmvn(self.cmvn_file)

    def output_size(self) -> int:
        return self.n_mels * self.lfr_m

    def forward(
        self,
        input: torch.Tensor,
        input_lengths,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            if self.upsacle_samples:
                waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.n_mels,
                frame_length=min(self.frame_length,waveform_length/self.fs*1000),
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type=self.window,
                sample_frequency=self.fs,
                snip_edges=self.snip_edges,
            )

            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            if self.cmvn is not None:
                mat = apply_cmvn(mat, self.cmvn)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        if batch_size == 1:
            feats_pad = feats[0][None, :, :]
        else:
            feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        return feats_pad, feats_lens

    def forward_fbank(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.n_mels,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type=self.window,
                sample_frequency=self.fs,
            )

            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        return feats_pad, feats_lens

    def forward_lfr_cmvn(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            mat = input[i, : input_lengths[i], :]
            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            if self.cmvn is not None:
                mat = apply_cmvn(mat, self.cmvn)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        return feats_pad, feats_lens


@tables.register("frontend_classes", "WavFrontendOnline")
class WavFrontendOnline(nn.Module):
    """Conventional frontend structure for streaming ASR/VAD."""

    def __init__(
        self,
        cmvn_file: str = None,
        fs: int = 16000,
        window: str = "hamming",
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
        **kwargs,
    ):
        super().__init__()
        self.fs = fs
        self.window = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.frame_sample_length = int(self.frame_length * self.fs / 1000)
        self.frame_shift_sample_length = int(self.frame_shift * self.fs / 1000)
        self.filter_length_min = filter_length_min
        self.filter_length_max = filter_length_max
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file
        self.dither = dither
        self.snip_edges = snip_edges
        self.upsacle_samples = upsacle_samples
        # self.waveforms = None
        # self.reserve_waveforms = None
        # self.fbanks = None
        # self.fbanks_lens = None
        self.cmvn = None if self.cmvn_file is None else load_cmvn(self.cmvn_file)
        # self.input_cache = None
        # self.lfr_splice_cache = []

    def output_size(self) -> int:
        return self.n_mels * self.lfr_m

    @staticmethod
    def apply_cmvn(inputs: torch.Tensor, cmvn: torch.Tensor) -> torch.Tensor:
        """
        Apply CMVN with mvn data
        """

        device = inputs.device
        dtype = inputs.dtype
        frame, dim = inputs.shape

        means = np.tile(cmvn[0:1, :dim], (frame, 1))
        vars = np.tile(cmvn[1:2, :dim], (frame, 1))
        inputs += torch.from_numpy(means).type(dtype).to(device)
        inputs *= torch.from_numpy(vars).type(dtype).to(device)

        return inputs.type(torch.float32)

    @staticmethod
    def apply_lfr(
        inputs: torch.Tensor, lfr_m: int, lfr_n: int, is_final: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Apply lfr with data
        """

        LFR_inputs = []
        # inputs = torch.vstack((inputs_lfr_cache, inputs))
        T = inputs.shape[0]  # include the right context
        T_lfr = int(
            np.ceil((T - (lfr_m - 1) // 2) / lfr_n)
        )  # minus the right context: (lfr_m - 1) // 2
        splice_idx = T_lfr
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).view(1, -1))
            else:  # process last LFR frame
                if is_final:
                    num_padding = lfr_m - (T - i * lfr_n)
                    frame = (inputs[i * lfr_n :]).view(-1)
                    for _ in range(num_padding):
                        frame = torch.hstack((frame, inputs[-1]))
                    LFR_inputs.append(frame)
                else:
                    # update splice_idx and break the circle
                    splice_idx = i
                    break
        splice_idx = min(T - 1, splice_idx * lfr_n)
        lfr_splice_cache = inputs[splice_idx:, :]
        LFR_outputs = torch.vstack(LFR_inputs)
        return LFR_outputs.type(torch.float32), lfr_splice_cache, splice_idx

    @staticmethod
    def compute_frame_num(
        sample_length: int, frame_sample_length: int, frame_shift_sample_length: int
    ) -> int:
        frame_num = int((sample_length - frame_sample_length) / frame_shift_sample_length + 1)
        return frame_num if frame_num >= 1 and sample_length >= frame_sample_length else 0

    def forward_fbank(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        cache: dict = {},
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)

        input = torch.cat((cache["input_cache"], input), dim=1)
        frame_num = self.compute_frame_num(
            input.shape[-1], self.frame_sample_length, self.frame_shift_sample_length
        )
        # update self.in_cache
        cache["input_cache"] = input[
            :, -(input.shape[-1] - frame_num * self.frame_shift_sample_length) :
        ]
        waveforms = torch.empty(0)
        feats_pad = torch.empty(0)
        feats_lens = torch.empty(0)
        if frame_num:
            waveforms = []
            feats = []
            feats_lens = []
            for i in range(batch_size):
                waveform = input[i]
                # we need accurate wave samples that used for fbank extracting
                waveforms.append(
                    waveform[
                        : (
                            (frame_num - 1) * self.frame_shift_sample_length
                            + self.frame_sample_length
                        )
                    ]
                )
                waveform = waveform * (1 << 15)
                waveform = waveform.unsqueeze(0)
                mat = kaldi.fbank(
                    waveform,
                    num_mel_bins=self.n_mels,
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift,
                    dither=self.dither,
                    energy_floor=0.0,
                    window_type=self.window,
                    sample_frequency=self.fs,
                )

                feat_length = mat.size(0)
                feats.append(mat)
                feats_lens.append(feat_length)

            waveforms = torch.stack(waveforms)
            feats_lens = torch.as_tensor(feats_lens)
            feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        cache["fbanks"] = feats_pad
        cache["fbanks_lens"] = copy.deepcopy(feats_lens)
        return waveforms, feats_pad, feats_lens

    def forward_lfr_cmvn(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        is_final: bool = False,
        cache: dict = {},
        **kwargs,
    ):
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        lfr_splice_frame_idxs = []
        for i in range(batch_size):
            mat = input[i, : input_lengths[i], :]
            if self.lfr_m != 1 or self.lfr_n != 1:
                # update self.lfr_splice_cache in self.apply_lfr
                # mat, self.lfr_splice_cache[i], lfr_splice_frame_idx = self.apply_lfr(mat, self.lfr_m, self.lfr_n, self.lfr_splice_cache[i],
                mat, cache["lfr_splice_cache"][i], lfr_splice_frame_idx = self.apply_lfr(
                    mat, self.lfr_m, self.lfr_n, is_final
                )
            if self.cmvn_file is not None:
                mat = self.apply_cmvn(mat, self.cmvn)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)
            lfr_splice_frame_idxs.append(lfr_splice_frame_idx)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        lfr_splice_frame_idxs = torch.as_tensor(lfr_splice_frame_idxs)
        return feats_pad, feats_lens, lfr_splice_frame_idxs

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor, **kwargs):
        is_final = kwargs.get("is_final", False)
        cache = kwargs.get("cache", {})
        if len(cache) == 0:
            self.init_cache(cache)

        batch_size = input.shape[0]
        assert (
            batch_size == 1
        ), "we support to extract feature online only when the batch size is equal to 1 now"

        waveforms, feats, feats_lengths = self.forward_fbank(
            input, input_lengths, cache=cache
        )  # input shape: B T D

        if feats.shape[0]:

            cache["waveforms"] = torch.cat((cache["reserve_waveforms"], waveforms), dim=1)

            if not cache["lfr_splice_cache"]:  # 初始化splice_cache
                for i in range(batch_size):
                    cache["lfr_splice_cache"].append(
                        feats[i][0, :].unsqueeze(dim=0).repeat((self.lfr_m - 1) // 2, 1)
                    )
            # need the number of the input frames + self.lfr_splice_cache[0].shape[0] is greater than self.lfr_m
            if feats_lengths[0] + cache["lfr_splice_cache"][0].shape[0] >= self.lfr_m:
                lfr_splice_cache_tensor = torch.stack(cache["lfr_splice_cache"])  # B T D
                feats = torch.cat((lfr_splice_cache_tensor, feats), dim=1)
                feats_lengths += lfr_splice_cache_tensor[0].shape[0]
                frame_from_waveforms = int(
                    (cache["waveforms"].shape[1] - self.frame_sample_length)
                    / self.frame_shift_sample_length
                    + 1
                )
                minus_frame = (
                    (self.lfr_m - 1) // 2 if cache["reserve_waveforms"].numel() == 0 else 0
                )
                feats, feats_lengths, lfr_splice_frame_idxs = self.forward_lfr_cmvn(
                    feats, feats_lengths, is_final, cache=cache
                )
                if self.lfr_m == 1:
                    cache["reserve_waveforms"] = torch.empty(0)
                else:
                    reserve_frame_idx = lfr_splice_frame_idxs[0] - minus_frame
                    # print('reserve_frame_idx:  ' + str(reserve_frame_idx))
                    # print('frame_frame:  ' + str(frame_from_waveforms))
                    cache["reserve_waveforms"] = cache["waveforms"][
                        :,
                        reserve_frame_idx
                        * self.frame_shift_sample_length : frame_from_waveforms
                        * self.frame_shift_sample_length,
                    ]
                    sample_length = (
                        frame_from_waveforms - 1
                    ) * self.frame_shift_sample_length + self.frame_sample_length
                    cache["waveforms"] = cache["waveforms"][:, :sample_length]
            else:
                # update self.reserve_waveforms and self.lfr_splice_cache
                cache["reserve_waveforms"] = cache["waveforms"][
                    :, : -(self.frame_sample_length - self.frame_shift_sample_length)
                ]
                for i in range(batch_size):
                    cache["lfr_splice_cache"][i] = torch.cat(
                        (cache["lfr_splice_cache"][i], feats[i]), dim=0
                    )
                return torch.empty(0), feats_lengths
        else:
            if is_final:
                cache["waveforms"] = (
                    waveforms
                    if cache["reserve_waveforms"].numel() == 0
                    else cache["reserve_waveforms"]
                )
                feats = torch.stack(cache["lfr_splice_cache"])
                feats_lengths = torch.zeros(batch_size, dtype=torch.int) + feats.shape[1]
                feats, feats_lengths, _ = self.forward_lfr_cmvn(
                    feats, feats_lengths, is_final, cache=cache
                )
        # if is_final:
        #     self.init_cache(cache)
        return feats, feats_lengths

    def init_cache(self, cache: dict = {}):
        cache["reserve_waveforms"] = torch.empty(0)
        cache["input_cache"] = torch.empty(0)
        cache["lfr_splice_cache"] = []
        cache["waveforms"] = None
        cache["fbanks"] = None
        cache["fbanks_lens"] = None
        return cache


class WavFrontendMel23(nn.Module):
    """Conventional frontend structure for ASR."""

    def __init__(
        self,
        fs: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 1,
        lfr_n: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.fs = fs
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.n_mels = 23

    def output_size(self) -> int:
        return self.n_mels * (2 * self.lfr_m + 1)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            waveform = waveform.numpy()
            mat = eend_ola_feature.stft(waveform, self.frame_length, self.frame_shift)
            mat = eend_ola_feature.transform(mat)
            mat = eend_ola_feature.splice(mat, context_size=self.lfr_m)
            mat = mat[:: self.lfr_n]
            mat = torch.from_numpy(mat)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        return feats_pad, feats_lens
