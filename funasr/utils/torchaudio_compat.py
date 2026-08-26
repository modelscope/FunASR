# Copyright (c) community contributors.
# Shared guard for the optional ``torchaudio`` dependency.
#
# FunASR's inference path no longer hard-depends on torchaudio: fbank feature
# extraction falls back to kaldi-native-fbank (see :mod:`funasr.utils.fbank`)
# and audio decoding falls back to soundfile/librosa. A few operations still
# require torchaudio; call :func:`require_torchaudio` at their entry point so a
# missing torchaudio fails with an actionable error instead of a bare
# ``AttributeError`` (or, worse, a silently wrong result).
import importlib.util

_INSTALL_HINT = (
    "Install a torchaudio build matching your torch version, e.g. "
    "``pip install torchaudio``. Some platforms -- notably Ascend NPU "
    "(aarch64) servers -- ship no matching torchaudio wheel; there, prefer a "
    "FunASR path that does not need torchaudio."
)


def torchaudio_available() -> bool:
    """Return ``True`` if torchaudio is importable."""
    return importlib.util.find_spec("torchaudio") is not None


def require_torchaudio(feature: str = "This operation"):
    """Return the torchaudio module, or raise an actionable ``ImportError``.

    Args:
        feature: Human-readable name of the operation that needs torchaudio,
            used in the error message.
    """
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires torchaudio, which is not installed. {_INSTALL_HINT}"
        ) from exc
    return torchaudio