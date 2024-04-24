# -*- encoding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Tuple, Union

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
