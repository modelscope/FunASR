"""Regression tests for SenseVoice rich transcription postprocessing."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PYTHON = REPO_ROOT / "runtime" / "python"
POSTPROCESS_UTILS = [
    REPO_ROOT / "funasr" / "utils" / "postprocess_utils.py",
    RUNTIME_PYTHON / "onnxruntime" / "funasr_onnx" / "utils" / "postprocess_utils.py",
    RUNTIME_PYTHON / "libtorch" / "funasr_torch" / "utils" / "postprocess_utils.py",
]


def _load(path):
    spec = importlib.util.spec_from_file_location(f"_postprocess_utils_{path.parts[-3]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", POSTPROCESS_UTILS, ids=lambda path: path.parts[-3])
@pytest.mark.parametrize(
    "event, emoji",
    [("<|Cough|>", "😷"), ("<|Sneeze|>", "🤧")],
)
def test_cough_and_sneeze_events_render_distinct_emojis(path, event, emoji):
    module = _load(path)

    text = module.rich_transcription_postprocess(f"<|en|><|NEUTRAL|>{event}<|woitn|>hello")

    assert text == f"{emoji}hello"
