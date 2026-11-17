import importlib.util
from pathlib import Path
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("whisper")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frontend_class(monkeypatch):
    # Isolate registration to avoid loading unrelated ASR models.
    registry = types.ModuleType("funasr.register")
    registry.tables = types.SimpleNamespace(register=lambda *args: lambda cls: cls)
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "funasr", types.ModuleType("funasr"))
        context.setitem(sys.modules, "funasr.register", registry)
        spec = importlib.util.spec_from_file_location(
            "whisper_frontend_batch_test", ROOT / "funasr/frontends/whisper_frontend.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.WhisperFrontend


@pytest.mark.parametrize(
    "lengths",
    [(1601, 4321), (4321, 1601), (3200, 3200), (3200,)],
    ids=["short-first", "long-first", "equal-lengths", "single-item"],
)
@pytest.mark.parametrize("do_pad_trim", [False, True])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize("length_type", [list, torch.tensor], ids=["list", "tensor"])
def test_batch_preserves_each_input_length(
    frontend_class, lengths, do_pad_trim, permute, length_type
):
    frontend = frontend_class(do_pad_trim=do_pad_trim, permute=permute)
    waveforms = torch.zeros(len(lengths), max(lengths))
    for i, length in enumerate(lengths):
        time = torch.arange(length, dtype=torch.float32) / frontend.fs
        waveforms[i, :length] = 0.1 * torch.sin(2 * torch.pi * (440 + i * 110) * time)
    input_lengths = length_type(lengths)

    features, feature_lengths = frontend(waveforms, input_lengths)

    expected_lengths = torch.tensor(lengths) // frontend.hop_length
    torch.testing.assert_close(feature_lengths, expected_lengths)
    samples = frontend.pad_samples if do_pad_trim else max(lengths)
    frames = samples // frontend.hop_length
    expected_shape = (len(lengths), frontend.n_mels, frames)
    if permute:
        expected_shape = (len(lengths), frames, frontend.n_mels)
    assert features.shape == expected_shape
    assert torch.isfinite(features).all()

    for i in range(len(lengths)):
        single_features, single_lengths = frontend(
            waveforms[i : i + 1], input_lengths[i : i + 1]
        )
        torch.testing.assert_close(features[i], single_features[0])
        torch.testing.assert_close(feature_lengths[i], single_lengths[0])
