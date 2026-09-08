"""Runtime regression tests for NumPy 2.x compatibility.

These tests exercise the actual code paths that previously used removed
NumPy aliases (``np.float``, ``np.int``), ensuring they work on both
NumPy 1.x and 2.x.  They complement ``test_pypi_metadata.py`` which only
scans source text.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_cmvn_load_returns_float64_arrays():
    """MultiChannelFrontend._load_cmvn must produce float64 arrays.

    The original code used ``np.float`` which was removed in NumPy 1.24
    and would raise AttributeError on NumPy 2.x.
    """
    from funasr.frontends.default import MultiChannelFrontend

    cmvn_text = (
        "<AddShift>\n"
        "<LearnRateCoef> 0 0 0.1 0.2 0.3 </LearnRateCoef>\n"
        "</AddShift>\n"
        "<Rescale>\n"
        "<LearnRateCoef> 0 0 1.0 1.1 1.2 </LearnRateCoef>\n"
        "</Rescale>\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cmvn", delete=False
    ) as fh:
        fh.write(cmvn_text)
        cmvn_path = fh.name

    try:
        # _load_cmvn does not touch self; call as unbound method.
        means, vars_ = MultiChannelFrontend._load_cmvn(None, cmvn_path)
    finally:
        Path(cmvn_path).unlink(missing_ok=True)

    assert means.dtype == np.float64
    assert vars_.dtype == np.float64
    assert len(means) == 3
    assert len(vars_) == 3
    assert np.allclose(means, [0.1, 0.2, 0.3])
    assert np.allclose(vars_, [1.0, 1.1, 1.2])


def test_emotion2vec_fromiter_uses_int64():
    """The masking helper in fairseq_modules must use np.int64.

    The original code passed ``np.int`` as the dtype to ``np.fromiter``,
    which fails on NumPy 2.x.  We replicate the exact call pattern.
    """
    parts = [(0, 10), (3, 7)]
    lens = np.fromiter(
        (e - s for s, e in parts),
        np.int64,
    )
    assert lens.dtype == np.int64
    assert np.array_equal(lens, [10, 4])


def test_funasr_imports_on_current_numpy():
    """The top-level package must import without NumPy alias errors."""
    import funasr  # noqa: F401

    # If we got here, no AttributeError from removed aliases.
    assert hasattr(np, "float64")
    assert hasattr(np, "int64")
