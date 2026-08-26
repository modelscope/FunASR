# Copyright (c) community contributors.
# A minimal drop-in fbank backend that lets FunASR run without torchaudio.
#
# torchaudio has no wheel matching the torch build on Ascend NPU (aarch64)
# servers, where torch is a Huawei-forked nightly (e.g. 2.14.0a0) whose version
# number is out of step with upstream PyTorch releases. This module falls back
# to kaldi-native-fbank -- a standalone C++ Kaldi fbank implementation with no
# torch dependency that ships manylinux aarch64 wheels.
#
# torchaudio.compliance.kaldi.fbank is only used on the CPU for feature
# extraction; the fallback matches its signature so call sites only need to
# change a single import line:
#     from funasr.utils import fbank as kaldi
import torch
import numpy as np

try:
    import torchaudio.compliance.kaldi as _torchaudio_kaldi

    _HAS_TORCHAUDIO = True
except Exception:  # noqa: BLE001 -- fall back to kaldi-native-fbank
    _HAS_TORCHAUDIO = False
    import kaldi_native_fbank as _knf


def _fbank_knf(
    waveform: torch.Tensor,
    num_mel_bins: int = 23,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    window_type: str = "povey",
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    raw_energy: bool = True,
    use_energy: bool = False,
    use_log_fbank: bool = True,
    use_power: bool = True,
    htk_compat: bool = False,
    preemphasis_coefficient: float = 0.97,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """kaldi-native-fbank implementation of :func:`torchaudio.compliance.kaldi.fbank`."""
    opts = _knf.FbankOptions()
    opts.frame_opts.samp_freq = float(sample_frequency)
    opts.frame_opts.frame_length_ms = float(frame_length)
    opts.frame_opts.frame_shift_ms = float(frame_shift)
    opts.frame_opts.dither = float(dither)
    opts.frame_opts.snip_edges = bool(snip_edges)
    opts.frame_opts.window_type = str(window_type)
    opts.frame_opts.preemph_coeff = float(preemphasis_coefficient)
    opts.frame_opts.remove_dc_offset = bool(remove_dc_offset)
    opts.frame_opts.round_to_power_of_two = bool(round_to_power_of_two)
    opts.mel_opts.num_bins = int(num_mel_bins)
    opts.mel_opts.low_freq = float(low_freq)
    opts.mel_opts.high_freq = float(high_freq)
    opts.energy_floor = float(energy_floor)
    opts.raw_energy = bool(raw_energy)
    opts.use_energy = bool(use_energy)
    opts.use_log_fbank = bool(use_log_fbank)
    opts.use_power = bool(use_power)
    opts.htk_compat = bool(htk_compat)

    fb = _knf.OnlineFbank(opts)
    wav = waveform.detach().cpu().numpy()
    while wav.ndim > 1:
        wav = wav.squeeze(0)
    wav = wav.astype(np.float32)
    fb.accept_waveform(int(sample_frequency), wav)
    fb.input_finished()
    n = fb.num_frames_ready
    if n == 0:
        return torch.zeros((0, int(num_mel_bins)), dtype=torch.float32)
    frames = np.stack([fb.get_frame(i) for i in range(n)])
    return torch.from_numpy(frames)


def fbank(waveform: torch.Tensor, **kwargs) -> torch.Tensor:
    """Drop-in replacement for :func:`torchaudio.compliance.kaldi.fbank`."""
    if _HAS_TORCHAUDIO:
        return _torchaudio_kaldi.fbank(waveform, **kwargs)
    return _fbank_knf(waveform, **kwargs)