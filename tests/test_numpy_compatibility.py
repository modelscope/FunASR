"""Exercise FunASR's NumPy boundaries without downloading model weights."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cmvn_path(tmp_path):
    path = tmp_path / "frontend.cmvn"
    path.write_text(
        "<AddShift>\n<LearnRateCoef> 0 0 "
        + " ".join(["0.1"] * 8)
        + " </LearnRateCoef>\n</AddShift>\n"
        + "<Rescale>\n<LearnRateCoef> 0 0 "
        + " ".join(["1.25"] * 8)
        + " </LearnRateCoef>\n</Rescale>\n",
        encoding="utf-8",
    )
    return path


def test_cmvn_load_returns_float64_arrays(cmvn_path):
    from funasr.frontends.default import MultiChannelFrontend

    means, scales = MultiChannelFrontend._load_cmvn(None, cmvn_path)
    assert means.dtype == scales.dtype == np.float64
    np.testing.assert_array_equal(means, np.full(8, 0.1, dtype=np.float64))
    np.testing.assert_array_equal(scales, np.full(8, 1.25, dtype=np.float64))


def test_cmvn_readers_skip_blank_lines(tmp_path):
    from funasr.frontends.default import MultiChannelFrontend
    from funasr.frontends.wav_frontend import load_cmvn

    path = tmp_path / "blank_lines.cmvn"
    path.write_text(
        "\n<AddShift>\n \t \n<LearnRateCoef> 0 0 "
        + " ".join(["0.1"] * 8)
        + " </LearnRateCoef>\n</AddShift>\n\n"
        + "<Rescale>\n<LearnRateCoef> 0 0 "
        + " ".join(["1.25"] * 8)
        + " </LearnRateCoef>\n</Rescale>\n\n",
        encoding="utf-8",
    )
    means, scales = MultiChannelFrontend._load_cmvn(None, path)
    np.testing.assert_array_equal(means, np.full(8, 0.1, dtype=np.float64))
    np.testing.assert_array_equal(scales, np.full(8, 1.25, dtype=np.float64))
    cmvn = load_cmvn(str(path))
    assert cmvn.shape == (2, 8)
    torch.testing.assert_close(cmvn[0], torch.full((8,), 0.1))
    torch.testing.assert_close(cmvn[1], torch.full((8,), 1.25))


def test_cmvn_readers_reject_files_without_statistics(tmp_path):
    from funasr.frontends.default import MultiChannelFrontend
    from funasr.frontends.wav_frontend import load_cmvn

    # Kaldi also emits the whole nnet on a single line; the readers walk lines,
    # so nothing matches and empty statistics used to be returned silently.
    compact = tmp_path / "compact.cmvn"
    compact.write_text(
        "<Nnet> <AddShift> <LearnRateCoef> 1 [ -1.0 -2.0 ] </LearnRateCoef> "
        "<Rescale> <LearnRateCoef> 1 [ 0.5 0.5 ] </Rescale> </Nnet>\n",
        encoding="utf-8",
    )
    truncated = tmp_path / "truncated.cmvn"
    truncated.write_text("<AddShift> 2 2\n", encoding="utf-8")
    for path in (compact, truncated):
        with pytest.raises(ValueError) as excinfo:
            MultiChannelFrontend._load_cmvn(None, path)
        assert str(path) in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            load_cmvn(str(path))
        assert str(path) in str(excinfo.value)


def test_cmvn_reader_loads_the_am_mvn_shipped_by_the_runtime():
    from funasr.frontends.wav_frontend import load_cmvn

    path = Path(__file__).resolve().parents[1] / (
        "runtime/triton_gpu/model_repo_sense_voice_small/feature_extractor/am.mvn"
    )
    cmvn = load_cmvn(str(path))
    assert cmvn.shape == (2, 560) and torch.isfinite(cmvn).all()


def test_frontend_applies_cmvn_and_preserves_padding(cmvn_path):
    from funasr.frontends.default import MultiChannelFrontend

    kwargs = dict(
        fs=16000,
        n_fft=512,
        frame_length=25,
        frame_shift=10,
        n_mels=8,
        center=False,
        mc=False,
    )
    plain = MultiChannelFrontend(**kwargs).eval()
    normalized = MultiChannelFrontend(**kwargs, cmvn_file=str(cmvn_path)).eval()
    # Initialization crosses the NumPy -> Torch boundary in production.
    assert normalized.mean.dtype == normalized.std.dtype == torch.float64
    np.testing.assert_array_equal(normalized.mean.numpy(), np.full(8, 0.1))

    wave = np.sin(2 * np.pi * 440 * np.arange(1600) / 16000).astype(np.float32)
    samples = torch.from_numpy(np.stack([wave, wave]))
    samples[1, 1200:] = 0
    lengths = torch.tensor([1600, 1200])
    with torch.no_grad():
        before, before_lengths = plain(samples, lengths)
        after, after_lengths = normalized(samples, lengths)
    torch.testing.assert_close(after_lengths, before_lengths)
    assert after.shape == before.shape and after.shape[-1] == 8
    assert torch.isfinite(after).all()
    for row, length in enumerate(after_lengths.tolist()):
        assert length > 0
        torch.testing.assert_close(
            after[row, :length], (before[row, :length] + 0.1) * 1.25
        )
        assert torch.count_nonzero(after[row, length:]) == 0


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_emotion2vec_no_overlap_respects_padding_and_spacing(seed):
    from funasr.models.emotion2vec.fairseq_modules import compute_mask_indices

    padding = torch.zeros((2, 80), dtype=torch.bool)
    padding[1, 53:] = True
    previous = np.random.get_state()
    try:
        np.random.seed(seed)
        mask = compute_mask_indices(
            (2, 80),
            padding,
            mask_prob=0.45,
            mask_length=4,
            no_overlap=True,
            min_space=2,
            require_same_masks=False,
        )
    finally:
        np.random.set_state(previous)
    assert mask.shape == (2, 80) and mask.dtype == np.bool_
    assert not mask[padding.numpy()].any()
    for row in mask:
        indices = np.flatnonzero(row)
        assert len(indices) > 0
        spans = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        assert all(len(span) == 4 for span in spans)
        assert all(
            right[0] - left[-1] - 1 >= 2 for left, right in zip(spans, spans[1:])
        )
    # Convert the real mask, not a duplicate np.fromiter expression.
    tensor_mask = torch.from_numpy(mask)
    assert tensor_mask.dtype == torch.bool
    np.testing.assert_array_equal(tensor_mask.numpy(), mask)


def test_relevant_modules_import_in_a_fresh_process():
    modules = [
        "funasr.frontends.default",
        "funasr.frontends.wav_frontend",
        "funasr.models.emotion2vec.fairseq_modules",
        "funasr.models.eend.utils.feature",
        "funasr.auto.auto_model",
    ]
    code = """
import importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
names = json.loads(sys.argv[2])
loaded = {}
for name in names:
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    assert path.is_relative_to(root), (name, str(path))
    loaded[name] = str(path)
import funasr
errors = funasr.get_import_errors()
assert not {name: errors[name] for name in names if name in errors}, errors
print('NUMPY_IMPORT_RESULT=' + json.dumps(loaded))
"""
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"})
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), json.dumps(modules)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = next(
        line.removeprefix("NUMPY_IMPORT_RESULT=")
        for line in result.stdout.splitlines()
        if line.startswith("NUMPY_IMPORT_RESULT=")
    )
    assert set(json.loads(payload)) == set(modules)
