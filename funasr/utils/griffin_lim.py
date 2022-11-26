#!/usr/bin/env python3

"""Griffin-Lim related modules."""

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

from distutils.version import LooseVersion
from functools import partial
from typeguard import check_argument_types
from typing import Optional

import librosa
import numpy as np
import torch

EPS = 1e-10


def logmel2linear(
    lmspc: np.ndarray,
    fs: int,
    n_fft: int,
    n_mels: int,
    fmin: int = None,
    fmax: int = None,
) -> np.ndarray:
    """Convert log Mel filterbank to linear spectrogram.

    Args:
        lmspc: Log Mel filterbank (T, n_mels).
        fs: Sampling frequency.
        n_fft: The number of FFT points.
        n_mels: The number of mel basis.
        f_min: Minimum frequency to analyze.
        f_max: Maximum frequency to analyze.

    Returns:
        Linear spectrogram (T, n_fft // 2 + 1).

    """
    assert lmspc.shape[1] == n_mels
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax
    mspc = np.power(10.0, lmspc)
    mel_basis = librosa.filters.mel(
        sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    inv_mel_basis = np.linalg.pinv(mel_basis)
    return np.maximum(EPS, np.dot(inv_mel_basis, mspc.T).T)


def griffin_lim(
    spc: np.ndarray,
    n_fft: int,
    n_shift: int,
    win_length: int = None,
    window: Optional[str] = "hann",
    n_iter: Optional[int] = 32,
) -> np.ndarray:
    """Convert linear spectrogram into waveform using Griffin-Lim.

    Args:
        spc: Linear spectrogram (T, n_fft // 2 + 1).
        n_fft: The number of FFT points.
        n_shift: Shift size in points.
        win_length: Window length in points.
        window: Window function type.
        n_iter: The number of iterations.

    Returns:
        Reconstructed waveform (N,).

    """
    # assert the size of input linear spectrogram
    assert spc.shape[1] == n_fft // 2 + 1

    if LooseVersion(librosa.__version__) >= LooseVersion("0.7.0"):
        # use librosa's fast Grriffin-Lim algorithm
        spc = np.abs(spc.T)
        y = librosa.griffinlim(
            S=spc,
            n_iter=n_iter,
            hop_length=n_shift,
            win_length=win_length,
            window=window,
            center=True if spc.shape[1] > 1 else False,
        )
    else:
        # use slower version of Grriffin-Lim algorithm
        logging.warning(
            "librosa version is old. use slow version of Grriffin-Lim algorithm."
            "if you want to use fast Griffin-Lim, please update librosa via "
            "`source ./path.sh && pip install librosa==0.7.0`."
        )
        cspc = np.abs(spc).astype(np.complex).T
        angles = np.exp(2j * np.pi * np.random.rand(*cspc.shape))
        y = librosa.istft(cspc * angles, n_shift, win_length, window=window)
        for i in range(n_iter):
            angles = np.exp(
                1j
                * np.angle(librosa.stft(y, n_fft, n_shift, win_length, window=window))
            )
            y = librosa.istft(cspc * angles, n_shift, win_length, window=window)

    return y


# TODO(kan-bayashi): write as torch.nn.Module
class Spectrogram2Waveform(object):
    """Spectrogram to waveform conversion module."""

    def __init__(
        self,
        n_fft: int,
        n_shift: int,
        fs: int = None,
        n_mels: int = None,
        win_length: int = None,
        window: Optional[str] = "hann",
        fmin: int = None,
        fmax: int = None,
        griffin_lim_iters: Optional[int] = 8,
    ):
        """Initialize module.

        Args:
            fs: Sampling frequency.
            n_fft: The number of FFT points.
            n_shift: Shift size in points.
            n_mels: The number of mel basis.
            win_length: Window length in points.
            window: Window function type.
            f_min: Minimum frequency to analyze.
            f_max: Maximum frequency to analyze.
            griffin_lim_iters: The number of iterations.

        """
        assert check_argument_types()
        self.fs = fs
        self.logmel2linear = (
            partial(
                logmel2linear, fs=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
            )
            if n_mels is not None
            else None
        )
        self.griffin_lim = partial(
            griffin_lim,
            n_fft=n_fft,
            n_shift=n_shift,
            win_length=win_length,
            window=window,
            n_iter=griffin_lim_iters,
        )
        self.params = dict(
            n_fft=n_fft,
            n_shift=n_shift,
            win_length=win_length,
            window=window,
            n_iter=griffin_lim_iters,
        )
        if n_mels is not None:
            self.params.update(fs=fs, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def __repr__(self):
        retval = f"{self.__class__.__name__}("
        for k, v in self.params.items():
            retval += f"{k}={v}, "
        retval += ")"
        return retval

    def __call__(self, spc: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to waveform.

        Args:
            spc: Log Mel filterbank (T_feats, n_mels)
                or linear spectrogram (T_feats, n_fft // 2 + 1).

        Returns:
            Tensor: Reconstructed waveform (T_wav,).

        """
        device = spc.device
        dtype = spc.dtype
        spc = spc.cpu().numpy()
        if self.logmel2linear is not None:
            spc = self.logmel2linear(spc)
        wav = self.griffin_lim(spc)
        return torch.tensor(wav).to(device=device, dtype=dtype)
