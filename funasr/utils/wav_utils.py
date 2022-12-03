# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os
from typing import Any, Dict, Union

import kaldiio
import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


def ndarray_resample(audio_in: np.ndarray,
                     fs_in: int = 16000,
                     fs_out: int = 16000) -> np.ndarray:
    audio_out = audio_in
    if fs_in != fs_out:
        audio_out = librosa.resample(audio_in, orig_sr=fs_in, target_sr=fs_out)
    return audio_out


def torch_resample(audio_in: torch.Tensor,
                   fs_in: int = 16000,
                   fs_out: int = 16000) -> torch.Tensor:
    audio_out = audio_in
    if fs_in != fs_out:
        audio_out = torchaudio.transforms.Resample(orig_freq=fs_in,
                                                   new_freq=fs_out)(audio_in)
    return audio_out


def extract_CMVN_featrures(mvn_file):
    """
    extract CMVN from cmvn.ark
    """

    if not os.path.exists(mvn_file):
        return None
    try:
        cmvn = kaldiio.load_mat(mvn_file)
        means = []
        variance = []

        for i in range(cmvn.shape[1] - 1):
            means.append(float(cmvn[0][i]))

        count = float(cmvn[0][-1])

        for i in range(cmvn.shape[1] - 1):
            variance.append(float(cmvn[1][i]))

        for i in range(len(means)):
            means[i] /= count
            variance[i] = variance[i] / count - means[i] * means[i]
            if variance[i] < 1.0e-20:
                variance[i] = 1.0e-20
            variance[i] = 1.0 / math.sqrt(variance[i])

        cmvn = np.array([means, variance])
        return cmvn
    except Exception:
        cmvn = extract_CMVN_features_txt(mvn_file)
        return cmvn


def extract_CMVN_features_txt(mvn_file):  # noqa
    with open(mvn_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    add_shift_list = []
    rescale_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == '<AddShift>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                add_shift_line = line_item[3:(len(line_item) - 1)]
                add_shift_list = list(add_shift_line)
                continue
        elif line_item[0] == '<Rescale>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                rescale_line = line_item[3:(len(line_item) - 1)]
                rescale_list = list(rescale_line)
                continue
    add_shift_list_f = [float(s) for s in add_shift_list]
    rescale_list_f = [float(s) for s in rescale_list]
    cmvn = np.array([add_shift_list_f, rescale_list_f])
    return cmvn


def build_LFR_features(inputs, m=7, n=6):  # noqa
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.

    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    left_padding = np.tile(inputs[0], ((m - 1) // 2, 1))
    inputs = np.vstack((left_padding, inputs))
    T = T + (m - 1) // 2
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
        else:  # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i * n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)


def compute_fbank(wav_file,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0,
                  is_pcm=False,
                  fs: Union[int, Dict[Any, int]] = 16000):
    audio_sr: int = 16000
    model_sr: int = 16000
    if isinstance(fs, int):
        model_sr = fs
        audio_sr = fs
    else:
        model_sr = fs['model_fs']
        audio_sr = fs['audio_fs']

    if is_pcm is True:
        # byte(PCM16) to float32, and resample
        value = wav_file
        middle_data = np.frombuffer(value, dtype=np.int16)
        middle_data = np.asarray(middle_data)
        if middle_data.dtype.kind not in 'iu':
            raise TypeError("'middle_data' must be an array of integers")
        dtype = np.dtype('float32')
        if dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        i = np.iinfo(middle_data.dtype)
        abs_max = 2**(i.bits - 1)
        offset = i.min + abs_max
        waveform = np.frombuffer(
            (middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
        waveform = ndarray_resample(waveform, audio_sr, model_sr)
        waveform = torch.from_numpy(waveform.reshape(1, -1))
    else:
        # load pcm from wav, and resample
        waveform, audio_sr = torchaudio.load(wav_file)
        waveform = waveform * (1 << 15)
        waveform = torch_resample(waveform, audio_sr, model_sr)

    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      energy_floor=0.0,
                      window_type='hamming',
                      sample_frequency=model_sr)

    input_feats = mat

    return input_feats
