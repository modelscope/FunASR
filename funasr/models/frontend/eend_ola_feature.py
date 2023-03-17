# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# This module is for computing audio features

import librosa
import numpy as np


def transform(Y, dtype=np.float32):
    Y = np.abs(Y)
    n_fft = 2 * (Y.shape[1] - 1)
    sr = 8000
    n_mels = 23
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
    Y = np.dot(Y ** 2, mel_basis.T)
    Y = np.log10(np.maximum(Y, 1e-10))
    mean = np.mean(Y, axis=0)
    Y = Y - mean
    return Y.astype(dtype)


def subsample(Y, T, subsampling=1):
    Y_ss = Y[::subsampling]
    T_ss = T[::subsampling]
    return Y_ss, T_ss


def splice(Y, context_size=0):
    Y_pad = np.pad(
        Y,
        [(context_size, context_size), (0, 0)],
        'constant')
    Y_spliced = np.lib.stride_tricks.as_strided(
        np.ascontiguousarray(Y_pad),
        (Y.shape[0], Y.shape[1] * (2 * context_size + 1)),
        (Y.itemsize * Y.shape[1], Y.itemsize), writeable=False)
    return Y_spliced


def stft(
        data,
        frame_size=1024,
        frame_shift=256):
    fft_size = 1 << (frame_size - 1).bit_length()
    if len(data) % frame_shift == 0:
        return librosa.stft(data, n_fft=fft_size, win_length=frame_size,
                            hop_length=frame_shift).T[:-1]
    else:
        return librosa.stft(data, n_fft=fft_size, win_length=frame_size,
                            hop_length=frame_shift).T