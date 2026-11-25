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


@pytest.mark.parametrize("path", POSTPROCESS_UTILS, ids=lambda path: path.parts[-3])
@pytest.mark.parametrize(
    "first, second, expected",
    [
        ("Hello world.", "How are you?", "Hello world. How are you?"),
        ("so I went to the", "store", "so I went to the store"),
        ("안녕하세요.", "감사합니다.", "안녕하세요. 감사합니다."),
        ("你好。", "再见。", "你好。再见。"),
        ("你好。", "Hello.", "你好。Hello."),
    ],
)
def test_vad_segments_keep_a_space_between_spaced_scripts(path, first, second, expected):
    module = _load(path)
    tags = "<|NEUTRAL|><|Speech|><|withitn|>"

    # AutoModel joins the tagged text of each VAD segment with a single space.
    text = module.rich_transcription_postprocess(f"<|en|>{tags}{first} <|en|>{tags}{second}")

    assert text == expected
