# Copyright (c) community contributors.
# A minimal fbank backend that lets FunASR run without torchaudio.
#
# torchaudio has no wheel matching the torch build on Ascend NPU (aarch64)
# servers, where torch is a Huawei-forked nightly (e.g. 2.14.0a0) whose version
# number is out of step with upstream PyTorch releases. This module forwards to
# torchaudio.compliance.kaldi.fbank when torchaudio is present and otherwise
# falls back to kaldi-native-fbank -- a standalone C++ Kaldi fbank
# implementation with no torch dependency that ships manylinux aarch64 wheels.
#
# torchaudio.compliance.kaldi.fbank is only used on the CPU for feature
# extraction. The fallback mirrors its signature so call sites only need to
# change a single import line:
#
#     from funasr.utils import fbank as kaldi
#
# Not a full drop-in: the fallback supports single-channel waveforms and the
# FbankOptions exposed by kaldi-native-fbank. torchaudio-only options such as
# ``subtract_mean``, ``min_duration``, VTLN warping or an explicit
# ``blackman_coeff`` are rejected with a clear error rather than silently
# ignored, and multi-channel waveforms without an explicit ``channel`` are
# rejected (kaldi-native-fbank is single-channel).
import torch
import numpy as np

try:
    import torchaudio.compliance.kaldi as _torchaudio_kaldi

    _HAS_TORCHAUDIO = True
except Exception:  # noqa: BLE001 -- torchaudio is optional here
    _HAS_TORCHAUDIO = False

_knf = None
if not _HAS_TORCHAUDIO:
    try:
        import kaldi_native_fbank as _knf
    except Exception:  # noqa: BLE001 -- reported lazily on first use
        _knf = None


def _knf_missing_error():
    return ImportError(
        "torchaudio is not installed and neither is the kaldi-native-fbank "
        "fallback backend. FunASR needs one fbank backend for feature "
        "extraction. Install either torchaudio (matching your torch version) "
        "or kaldi-native-fbank, e.g. `pip install kaldi-native-fbank` (or "
        "`pip install funasr[knf]`)."
    )


def _fbank_knf(
    waveform: torch.Tensor,
    blackman_coeff: float = 0.42,
    channel: int = -1,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    high_freq: float = 0.0,
    htk_compat: bool = False,
    low_freq: float = 20.0,
    min_duration: float = 0.0,
    num_mel_bins: int = 23,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    subtract_mean: bool = False,
    use_energy: bool = False,
    use_log_fbank: bool = True,
    use_power: bool = True,
    vtln_high: float = -500.0,
    vtln_low: float = 100.0,
    vtln_warp: float = 1.0,
    window_type: str = "povey",
) -> torch.Tensor:
    """kaldi-native-fbank implementation of ``torchaudio.compliance.kaldi.fbank``."""
    if _knf is None:
        raise _knf_missing_error()

    # torchaudio options the kaldi-native-fbank fallback does not implement.
    # Reject them explicitly instead of silently dropping them.
    unsupported = {}
    if blackman_coeff != 0.42:
        unsupported["blackman_coeff"] = blackman_coeff
    if min_duration != 0.0:
        unsupported["min_duration"] = min_duration
    if subtract_mean:
        unsupported["subtract_mean"] = subtract_mean
    if vtln_warp != 1.0:
        unsupported["vtln_warp"] = vtln_warp
    if vtln_low != 100.0:
        unsupported["vtln_low"] = vtln_low
    if vtln_high != -500.0:
        unsupported["vtln_high"] = vtln_high
    if unsupported:
        raise NotImplementedError(
            "The kaldi-native-fbank fallback does not support these "
            f"torchaudio fbank options: {sorted(unsupported)}. Install "
            "torchaudio to use them, or drop these options."
        )

    # Channel / waveform-shape handling, mirroring torchaudio semantics for the
    # single-channel case kaldi-native-fbank supports.
    if waveform.dim() == 2:
        if channel >= 0:
            waveform = waveform[channel]
        elif waveform.size(0) == 1:
            waveform = waveform[0]
        else:
            raise NotImplementedError(
                "The kaldi-native-fbank fallback extracts features from a "
                "single channel only. Pass `channel` to select one channel, or "
                "reduce the waveform to mono before calling fbank()."
            )
    elif waveform.dim() == 1:
        if channel > 0:
            raise ValueError(
                f"Invalid channel {channel} for a 1-D waveform; use channel=-1 or 0."
            )
    else:
        raise ValueError("waveform must be a 1-D or 2-D tensor")

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
    wav = waveform.detach().cpu().numpy().reshape(-1).astype(np.float32)
    fb.accept_waveform(int(sample_frequency), wav)
    fb.input_finished()
    n = fb.num_frames_ready
    if n == 0:
        return torch.zeros((0, int(num_mel_bins)), dtype=torch.float32)
    frames = np.stack([fb.get_frame(i) for i in range(n)])
    return torch.from_numpy(frames)


def fbank(waveform: torch.Tensor, **kwargs) -> torch.Tensor:
    """Drop-in replacement for :func:`torchaudio.compliance.kaldi.fbank`.

    Forwards to torchaudio when it is installed; otherwise falls back to
    kaldi-native-fbank, rejecting unsupported options explicitly (see module
    docstring).
    """
    if _HAS_TORCHAUDIO:
        return _torchaudio_kaldi.fbank(waveform, **kwargs)
    return _fbank_knf(waveform, **kwargs)