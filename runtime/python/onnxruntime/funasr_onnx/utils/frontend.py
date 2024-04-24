# -*- encoding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Tuple, Union
import copy

import numpy as np
import kaldi_native_fbank as knf

root_dir = Path(__file__).resolve().parent

logger_initialized = {}


class WavFrontend:
    """Conventional frontend structure for ASR."""

    def __init__(
        self,
        cmvn_file: str = None,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 1,
        lfr_n: int = 1,
        dither: float = 1.0,
        **kwargs,
    ) -> None:

        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = fs
        opts.frame_opts.dither = dither
        opts.frame_opts.window_type = window
        opts.frame_opts.frame_shift_ms = float(frame_shift)
        opts.frame_opts.frame_length_ms = float(frame_length)
        opts.mel_opts.num_bins = n_mels
        opts.energy_floor = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file

        if self.cmvn_file:
            self.cmvn = self.load_cmvn()
        self.fbank_fn = None
        self.fbank_beg_idx = 0
        self.reset_status()

    def fbank(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = waveform * (1 << 15)
        self.fbank_fn = knf.OnlineFbank(self.opts)
        self.fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        frames = self.fbank_fn.num_frames_ready
        mat = np.empty([frames, self.opts.mel_opts.num_bins])
        for i in range(frames):
            mat[i, :] = self.fbank_fn.get_frame(i)
        feat = mat.astype(np.float32)
        feat_len = np.array(mat.shape[0]).astype(np.int32)
        return feat, feat_len

    def fbank_online(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = waveform * (1 << 15)
        # self.fbank_fn = knf.OnlineFbank(self.opts)
        self.fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        frames = self.fbank_fn.num_frames_ready
        mat = np.empty([frames, self.opts.mel_opts.num_bins])
        for i in range(self.fbank_beg_idx, frames):
            mat[i, :] = self.fbank_fn.get_frame(i)
        # self.fbank_beg_idx += (frames-self.fbank_beg_idx)
        feat = mat.astype(np.float32)
        feat_len = np.array(mat.shape[0]).astype(np.int32)
        return feat, feat_len

    def reset_status(self):
        self.fbank_fn = knf.OnlineFbank(self.opts)
        self.fbank_beg_idx = 0

    def lfr_cmvn(self, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.lfr_m != 1 or self.lfr_n != 1:
            feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)

        if self.cmvn_file:
            feat = self.apply_cmvn(feat)

        feat_len = np.array(feat.shape[0]).astype(np.int32)
        return feat, feat_len

    @staticmethod
    def apply_lfr(inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
        LFR_inputs = []

        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))
        left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
        inputs = np.vstack((left_padding, inputs))
        T = T + (lfr_m - 1) // 2
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
            else:
                # process last LFR frame
                num_padding = lfr_m - (T - i * lfr_n)
                frame = inputs[i * lfr_n :].reshape(-1)
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))

                LFR_inputs.append(frame)
        LFR_outputs = np.vstack(LFR_inputs).astype(np.float32)
        return LFR_outputs

    def apply_cmvn(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply CMVN with mvn data
        """
        frame, dim = inputs.shape
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
        vars = np.tile(self.cmvn[1:2, :dim], (frame, 1))
        inputs = (inputs + means) * vars
        return inputs

    def load_cmvn(
        self,
    ) -> np.ndarray:
        with open(self.cmvn_file, "r", encoding="utf-8") as f:
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

        means = np.array(means_list).astype(np.float64)
        vars = np.array(vars_list).astype(np.float64)
        cmvn = np.array([means, vars])
        return cmvn


class WavFrontendOnline(WavFrontend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.fbank_fn = knf.OnlineFbank(self.opts)
        # add variables
        self.frame_sample_length = int(
            self.opts.frame_opts.frame_length_ms * self.opts.frame_opts.samp_freq / 1000
        )
        self.frame_shift_sample_length = int(
            self.opts.frame_opts.frame_shift_ms * self.opts.frame_opts.samp_freq / 1000
        )
        self.waveform = None
        self.reserve_waveforms = None
        self.input_cache = None
        self.lfr_splice_cache = []

    @staticmethod
    # inputs has catted the cache
    def apply_lfr(
        inputs: np.ndarray, lfr_m: int, lfr_n: int, is_final: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Apply lfr with data
        """

        LFR_inputs = []
        T = inputs.shape[0]  # include the right context
        T_lfr = int(
            np.ceil((T - (lfr_m - 1) // 2) / lfr_n)
        )  # minus the right context: (lfr_m - 1) // 2
        splice_idx = T_lfr
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
            else:  # process last LFR frame
                if is_final:
                    num_padding = lfr_m - (T - i * lfr_n)
                    frame = (inputs[i * lfr_n :]).reshape(-1)
                    for _ in range(num_padding):
                        frame = np.hstack((frame, inputs[-1]))
                    LFR_inputs.append(frame)
                else:
                    # update splice_idx and break the circle
                    splice_idx = i
                    break
        splice_idx = min(T - 1, splice_idx * lfr_n)
        lfr_splice_cache = inputs[splice_idx:, :]
        LFR_outputs = np.vstack(LFR_inputs)
        return LFR_outputs.astype(np.float32), lfr_splice_cache, splice_idx

    @staticmethod
    def compute_frame_num(
        sample_length: int, frame_sample_length: int, frame_shift_sample_length: int
    ) -> int:
        frame_num = int((sample_length - frame_sample_length) / frame_shift_sample_length + 1)
        return frame_num if frame_num >= 1 and sample_length >= frame_sample_length else 0

    def fbank(
        self, input: np.ndarray, input_lengths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.fbank_fn = knf.OnlineFbank(self.opts)
        batch_size = input.shape[0]
        if self.input_cache is None:
            self.input_cache = np.empty((batch_size, 0), dtype=np.float32)
        input = np.concatenate((self.input_cache, input), axis=1)
        frame_num = self.compute_frame_num(
            input.shape[-1], self.frame_sample_length, self.frame_shift_sample_length
        )
        # update self.in_cache
        self.input_cache = input[
            :, -(input.shape[-1] - frame_num * self.frame_shift_sample_length) :
        ]
        waveforms = np.empty(0, dtype=np.float32)
        feats_pad = np.empty(0, dtype=np.float32)
        feats_lens = np.empty(0, dtype=np.int32)
        if frame_num:
            waveforms = []
            feats = []
            feats_lens = []
            for i in range(batch_size):
                waveform = input[i]
                waveforms.append(
                    waveform[
                        : (
                            (frame_num - 1) * self.frame_shift_sample_length
                            + self.frame_sample_length
                        )
                    ]
                )
                waveform = waveform * (1 << 15)

                self.fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
                frames = self.fbank_fn.num_frames_ready
                mat = np.empty([frames, self.opts.mel_opts.num_bins])
                for i in range(frames):
                    mat[i, :] = self.fbank_fn.get_frame(i)
                feat = mat.astype(np.float32)
                feat_len = np.array(mat.shape[0]).astype(np.int32)
                feats.append(feat)
                feats_lens.append(feat_len)

            waveforms = np.stack(waveforms)
            feats_lens = np.array(feats_lens)
            feats_pad = np.array(feats)
        self.fbanks = feats_pad
        self.fbanks_lens = copy.deepcopy(feats_lens)
        return waveforms, feats_pad, feats_lens

    def get_fbank(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.fbanks, self.fbanks_lens

    def lfr_cmvn(
        self, input: np.ndarray, input_lengths: np.ndarray, is_final: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        batch_size = input.shape[0]
        feats = []
        feats_lens = []
        lfr_splice_frame_idxs = []
        for i in range(batch_size):
            mat = input[i, : input_lengths[i], :]
            lfr_splice_frame_idx = -1
            if self.lfr_m != 1 or self.lfr_n != 1:
                # update self.lfr_splice_cache in self.apply_lfr
                mat, self.lfr_splice_cache[i], lfr_splice_frame_idx = self.apply_lfr(
                    mat, self.lfr_m, self.lfr_n, is_final
                )
            if self.cmvn_file is not None:
                mat = self.apply_cmvn(mat)
            feat_length = mat.shape[0]
            feats.append(mat)
            feats_lens.append(feat_length)
            lfr_splice_frame_idxs.append(lfr_splice_frame_idx)

        feats_lens = np.array(feats_lens)
        feats_pad = np.array(feats)
        return feats_pad, feats_lens, lfr_splice_frame_idxs

    def extract_fbank(
        self, input: np.ndarray, input_lengths: np.ndarray, is_final: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = input.shape[0]
        assert (
            batch_size == 1
        ), "we support to extract feature online only when the batch size is equal to 1 now"
        waveforms, feats, feats_lengths = self.fbank(input, input_lengths)  # input shape: B T D
        if feats.shape[0]:
            self.waveforms = (
                waveforms
                if self.reserve_waveforms is None
                else np.concatenate((self.reserve_waveforms, waveforms), axis=1)
            )
            if not self.lfr_splice_cache:
                for i in range(batch_size):
                    self.lfr_splice_cache.append(
                        np.expand_dims(feats[i][0, :], axis=0).repeat((self.lfr_m - 1) // 2, axis=0)
                    )

            if feats_lengths[0] + self.lfr_splice_cache[0].shape[0] >= self.lfr_m:
                lfr_splice_cache_np = np.stack(self.lfr_splice_cache)  # B T D
                feats = np.concatenate((lfr_splice_cache_np, feats), axis=1)
                feats_lengths += lfr_splice_cache_np[0].shape[0]
                frame_from_waveforms = int(
                    (self.waveforms.shape[1] - self.frame_sample_length)
                    / self.frame_shift_sample_length
                    + 1
                )
                minus_frame = (self.lfr_m - 1) // 2 if self.reserve_waveforms is None else 0
                feats, feats_lengths, lfr_splice_frame_idxs = self.lfr_cmvn(
                    feats, feats_lengths, is_final
                )
                if self.lfr_m == 1:
                    self.reserve_waveforms = None
                else:
                    reserve_frame_idx = lfr_splice_frame_idxs[0] - minus_frame
                    # print('reserve_frame_idx:  ' + str(reserve_frame_idx))
                    # print('frame_frame:  ' + str(frame_from_waveforms))
                    self.reserve_waveforms = self.waveforms[
                        :,
                        reserve_frame_idx
                        * self.frame_shift_sample_length : frame_from_waveforms
                        * self.frame_shift_sample_length,
                    ]
                    sample_length = (
                        frame_from_waveforms - 1
                    ) * self.frame_shift_sample_length + self.frame_sample_length
                    self.waveforms = self.waveforms[:, :sample_length]
            else:
                # update self.reserve_waveforms and self.lfr_splice_cache
                self.reserve_waveforms = self.waveforms[
                    :, : -(self.frame_sample_length - self.frame_shift_sample_length)
                ]
                for i in range(batch_size):
                    self.lfr_splice_cache[i] = np.concatenate(
                        (self.lfr_splice_cache[i], feats[i]), axis=0
                    )
                return np.empty(0, dtype=np.float32), feats_lengths
        else:
            if is_final:
                self.waveforms = (
                    waveforms if self.reserve_waveforms is None else self.reserve_waveforms
                )
                feats = np.stack(self.lfr_splice_cache)
                feats_lengths = np.zeros(batch_size, dtype=np.int32) + feats.shape[1]
                feats, feats_lengths, _ = self.lfr_cmvn(feats, feats_lengths, is_final)
        if is_final:
            self.cache_reset()
        return feats, feats_lengths

    def get_waveforms(self):
        return self.waveforms

    def cache_reset(self):
        self.fbank_fn = knf.OnlineFbank(self.opts)
        self.reserve_waveforms = None
        self.input_cache = None
        self.lfr_splice_cache = []


def load_bytes(input):
    middle_data = np.frombuffer(input, dtype=np.int16)
    middle_data = np.asarray(middle_data)
    if middle_data.dtype.kind not in "iu":
        raise TypeError("'middle_data' must be an array of integers")
    dtype = np.dtype("float32")
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(middle_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
    return array


class SinusoidalPositionEncoderOnline:
    """Streaming Positional encoding."""

    def encode(self, positions: np.ndarray = None, depth: int = None, dtype: np.dtype = np.float32):
        batch_size = positions.shape[0]
        positions = positions.astype(dtype)
        log_timescale_increment = np.log(np.array([10000], dtype=dtype)) / (depth / 2 - 1)
        inv_timescales = np.exp(np.arange(depth / 2).astype(dtype) * (-log_timescale_increment))
        inv_timescales = np.reshape(inv_timescales, [batch_size, -1])
        scaled_time = np.reshape(positions, [1, -1, 1]) * np.reshape(inv_timescales, [1, 1, -1])
        encoding = np.concatenate((np.sin(scaled_time), np.cos(scaled_time)), axis=2)
        return encoding.astype(dtype)

    def forward(self, x, start_idx=0):
        batch_size, timesteps, input_dim = x.shape
        positions = np.arange(1, timesteps + 1 + start_idx)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype)

        return x + position_encoding[:, start_idx : start_idx + timesteps]


def test():
    path = "/nfs/zhifu.gzf/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav"
    import librosa

    cmvn_file = "/nfs/zhifu.gzf/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn"
    config_file = "/nfs/zhifu.gzf/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/config.yaml"
    from funasr.runtime.python.onnxruntime.rapid_paraformer.utils.utils import read_yaml

    config = read_yaml(config_file)
    waveform, _ = librosa.load(path, sr=None)
    frontend = WavFrontend(
        cmvn_file=cmvn_file,
        **config["frontend_conf"],
    )
    speech, _ = frontend.fbank_online(waveform)  # 1d, (sample,), numpy
    feat, feat_len = frontend.lfr_cmvn(
        speech
    )  # 2d, (frame, 450), np.float32 -> torch, torch.from_numpy(), dtype, (1, frame, 450)

    frontend.reset_status()  # clear cache
    return feat, feat_len


if __name__ == "__main__":
    test()
