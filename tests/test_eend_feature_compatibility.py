"""Check EEND feature values across the supported scientific-stack matrix."""

import numpy as np
import pytest
import torch


@pytest.fixture
def spectra():
    amplitudes = np.array([1, 10, 100, 1000], dtype=np.float32)
    return amplitudes[:, None] * np.full((4, 257), 1 + 1j, dtype=np.complex64)


@pytest.mark.parametrize("mode,bands", [("logmel", 40), ("logmel23", 23)])
def test_eend_logmel_preserves_power_ratios(spectra, mode, bands):
    from funasr.models.eend.utils.feature import transform

    result = transform(spectra, transform_type=mode)
    assert result.dtype == np.float32 and result.shape == (4, bands)
    assert np.isfinite(result).all()
    # A tenfold amplitude increase adds 2 to log10 power in every mel band.
    np.testing.assert_allclose(np.diff(result, axis=0), 2.0, atol=2e-6)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("logmel23_mn", [-3, -1, 1, 3]),
        ("logmel23_mvn", np.array([-3, -1, 1, 3]) / np.sqrt(5)),
        ("logmel23_swn", [-5, -3, -1, 1]),
    ],
)
def test_eend_normalization_values(spectra, mode, expected):
    from funasr.models.eend.utils.feature import transform

    original = spectra.copy()
    result = transform(spectra, transform_type=mode)
    assert result.dtype == np.float32
    np.testing.assert_allclose(
        result, np.repeat(np.asarray(expected)[:, None], 23, axis=1), atol=2e-6
    )
    np.testing.assert_array_equal(spectra, original)


def test_eend_ola_mean_normalization(spectra):
    from funasr.frontends.eend_ola_feature import transform

    result = transform(spectra)
    np.testing.assert_allclose(
        result, np.repeat(np.array([-3, -1, 1, 3])[:, None], 23, axis=1), atol=2e-6
    )
    assert result.dtype == np.float32


def test_mel23_waveform_frontend_returns_finite_padded_features():
    from funasr.frontends.wav_frontend import WavFrontendMel23

    wave = np.sin(2 * np.pi * 440 * np.arange(1600) / 8000).astype(np.float32)
    samples = torch.from_numpy(np.stack([wave, wave]))
    frontend = WavFrontendMel23(
        fs=8000, frame_length=256, frame_shift=80, lfr_m=0, lfr_n=1
    )
    features, lengths = frontend(samples, torch.tensor([1600, 1200]))
    assert features.shape == (2, 20, 23)
    assert lengths.tolist() == [20, 15]
    assert features.dtype == torch.float32 and torch.isfinite(features).all()
    assert torch.count_nonzero(features[1, 15:]) == 0
    assert torch.count_nonzero(features[0]) > 0
