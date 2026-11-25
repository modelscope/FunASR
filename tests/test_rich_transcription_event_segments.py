"""Keep event-only language segments consistent across shipped runtimes."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    ROOT / "funasr/utils/postprocess_utils.py",
    ROOT / "runtime/python/onnxruntime/funasr_onnx/utils/postprocess_utils.py",
    ROOT / "runtime/python/libtorch/funasr_torch/utils/postprocess_utils.py",
]


@pytest.fixture(params=MODULES, ids=["funasr", "onnx", "libtorch"])
def postprocess(request):
    spec = importlib.util.spec_from_file_location("_event_segments", request.param)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("event", ["<|Cough|>", "<|Sneeze|>", "<|Applause|>"])
@pytest.mark.parametrize("following_text", ["", "world"])
def test_repeated_event_only_segments_keep_following_text(postprocess, event, following_text):
    text = (
        f"<|en|><|NEUTRAL|>{event}<|woitn|>hello"
        f"<|zh|><|NEUTRAL|>{event}<|woitn|>"
        f"<|ja|><|NEUTRAL|>{event}<|woitn|>"
        f"<|en|><|NEUTRAL|><|Speech|><|woitn|>{following_text}"
    )

    result = postprocess.rich_transcription_postprocess(text)

    assert result == postprocess.event_dict[event] + " ".join(
        filter(None, ["hello", following_text])
    )


@pytest.mark.parametrize("event", ["<|Cough|>", "<|Sneeze|>", "<|Applause|>"])
def test_repeated_event_only_segments_without_speech(postprocess, event):
    text = f"<|en|><|NEUTRAL|>{event}<|zh|><|NEUTRAL|>{event}"

    assert postprocess.rich_transcription_postprocess(text) == postprocess.event_dict[event]


def test_distinct_event_only_segments_remain_visible(postprocess):
    text = "<|en|><|NEUTRAL|><|Applause|><|zh|><|NEUTRAL|><|Laughter|>"

    assert postprocess.rich_transcription_postprocess(text) == (
        postprocess.event_dict["<|Applause|>"] + postprocess.event_dict["<|Laughter|>"]
    )


@pytest.mark.parametrize("event", ["<|Cough|>", "<|Sneeze|>", "<|Applause|>"])
def test_repeated_event_with_text_across_three_segments(postprocess, event):
    text = (
        f"<|en|>{event}hello"
        f"<|en|>{event}world"
        f"<|en|>{event}again"
    )
    emoji = postprocess.event_dict[event]

    assert postprocess.rich_transcription_postprocess(text) == f"{emoji}hello world again"


def test_different_event_between_same_events_keeps_all_emojis(postprocess):
    text = (
        "<|en|><|Applause|>hello"
        "<|en|><|Laughter|>world"
        "<|en|><|Applause|>again"
    )
    applause = postprocess.event_dict["<|Applause|>"]
    laughter = postprocess.event_dict["<|Laughter|>"]

    assert postprocess.rich_transcription_postprocess(text) == (
        f"{applause}hello{laughter}world{applause}again"
    )
