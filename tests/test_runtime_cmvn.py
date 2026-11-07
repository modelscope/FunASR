"""Exercise the standalone runtime frontends without model weights or engines."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CMVN = "<AddShift> 2 2\n<LearnRateCoef> 1 [ 1 -2 ]\n<Rescale> 2 2\n<LearnRateCoef> 1 [ 0.5 2 ]\n"


@pytest.fixture(params=["onnxruntime/funasr_onnx", "libtorch/funasr_torch"])
def frontend_class(request):
    # Loading the frontend directly keeps these tests independent of the
    # optional inference engines and exercises each package's real constructor.
    path = ROOT / "runtime/python" / request.param / "utils/frontend.py"
    spec = importlib.util.spec_from_file_location("runtime_cmvn_frontend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WavFrontend


def test_valid_cmvn_preserves_statistics_and_normalized_features(frontend_class, tmp_path):
    path = tmp_path / "valid.mvn"
    path.write_text(CMVN, encoding="utf-8")
    frontend = frontend_class(cmvn_file=str(path))

    assert frontend.cmvn.dtype == np.float64
    np.testing.assert_array_equal(frontend.cmvn, [[1, -2], [0.5, 2]])
    features = np.array([[0, 2], [-2, 4]], dtype=np.float32)
    np.testing.assert_array_equal(frontend.apply_cmvn(features), [[0.5, 0], [-0.5, 4]])


@pytest.mark.parametrize("placement", ["leading", "trailing", "between_sections", "after_tags"])
def test_cmvn_accepts_blank_lines(frontend_class, tmp_path, placement):
    blank = " \t\n\n"
    if placement == "leading":
        content = blank + CMVN
    elif placement == "trailing":
        content = CMVN + blank
    elif placement == "between_sections":
        content = CMVN.replace("<Rescale>", blank + "<Rescale>")
    else:
        content = CMVN.replace("<LearnRateCoef>", blank + "<LearnRateCoef>")
    path = tmp_path / "blank lines.mvn"
    path.write_text(content, encoding="utf-8")

    frontend = frontend_class(cmvn_file=str(path))

    np.testing.assert_array_equal(frontend.cmvn, [[1, -2], [0.5, 2]])


@pytest.mark.parametrize(
    "content",
    [
        "",
        " \t\n",
        "<Nnet>\n</Nnet>\n",
        "<AddShift> 2 2\n",
        "<Rescale> 2 2\n",
        "<AddShift> 2 2\n<LearnRateCoef> 1 [ 1 -2 ]\n",
        "<Rescale> 2 2\n<LearnRateCoef> 1 [ 0.5 2 ]\n",
        CMVN.replace("\n", " "),
    ],
    ids=[
        "empty",
        "blank",
        "no_stats",
        "truncated_shift",
        "truncated_scale",
        "shift_only",
        "scale_only",
        "compact",
    ],
)
def test_incomplete_cmvn_fails_during_construction(frontend_class, tmp_path, content):
    path = tmp_path / "incomplete.mvn"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as error:
        frontend_class(cmvn_file=str(path))

    assert str(path) in str(error.value)


def test_shipped_cmvn_is_unchanged_by_blank_lines(frontend_class, tmp_path):
    shipped = ROOT / "runtime/triton_gpu/model_repo_sense_voice_small/feature_extractor/am.mvn"
    original = frontend_class(cmvn_file=str(shipped), lfr_m=7, lfr_n=6, dither=0.0)
    path = tmp_path / "shipped-with-blanks.mvn"
    path.write_text("\n" + "\n \t\n".join(shipped.read_text().splitlines()) + "\n\n")
    edited = frontend_class(cmvn_file=str(path), lfr_m=7, lfr_n=6, dither=0.0)

    assert original.cmvn.shape == (2, 560)
    assert np.isfinite(original.cmvn).all()
    np.testing.assert_array_equal(edited.cmvn, original.cmvn)
    # Real filterbank + LFR + CMVN on a deterministic waveform, not a model stub.
    waveform = np.sin(2 * np.pi * 440 * np.arange(1600) / 16000).astype(np.float32) * 0.1
    original_features, _ = original.fbank(waveform)
    edited_features, _ = edited.fbank(waveform)
    original_features, original_length = original.lfr_cmvn(original_features)
    edited_features, edited_length = edited.lfr_cmvn(edited_features)
    assert edited_length == original_length
    np.testing.assert_array_equal(edited_features, original_features)
