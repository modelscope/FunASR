"""Repetitive-garbage trimming must behave identically in both Nano runtimes.

``_clean_text`` removes repeated garbage with
``re.sub(r'(>.{2,8}?)\\1{3,}', '', text)``, which the non-streaming engine
(``inference_vllm_pipeline.py``) has always done.

The streaming engine's copy of the same function carried a raw ``0x01`` byte
where the backreference ``\\1`` belongs. The group reference therefore became a
control character, and the pattern could only match a literal control byte, so
the streaming engine kept the repetitive garbage that the pipeline engine
strips - for the same model, the same audio and the same ``_clean_text``
contract.

These tests import both production ``_clean_text`` implementations and require
them to agree. No weights, GPU or model artifact are needed.
"""

import pytest

from funasr.models.fun_asr_nano.inference_vllm_pipeline import (
    _clean_text as pipeline_clean_text,
)
from funasr.models.fun_asr_nano.inference_vllm_streaming import (
    _clean_text as streaming_clean_text,
)

# The pattern is '>' plus 2-8 characters, repeated at least four times in all.
REPETITIVE_CASES = [
    (">abc>abc>abc>abc", ""),
    (">你好世界>你好世界>你好世界>你好世界", ""),
    ("正常文本>重复片段>重复片段>重复片段>重复片段结尾", "正常文本结尾"),
]

# Inputs the repetitive-garbage rule must leave alone.
PLAIN_CASES = [
    "",
    "正常文本",
    "hello world",
    ">abc>abc>abc",  # three repeats: below the four-repeat threshold
    "a>a>a>a",  # the repeated group needs at least two characters
    "hello\x01world",  # a genuine control byte must not be treated as garbage
    "<|startofspeech|>你好",
]


@pytest.mark.parametrize("text,expected", REPETITIVE_CASES)
def test_streaming_trims_repetitive_garbage(text, expected):
    assert streaming_clean_text(text) == expected


@pytest.mark.parametrize("text,expected", REPETITIVE_CASES)
def test_streaming_matches_pipeline_on_repetitive_garbage(text, expected):
    assert streaming_clean_text(text) == pipeline_clean_text(text) == expected


@pytest.mark.parametrize("text", PLAIN_CASES)
def test_both_runtimes_agree_on_text_without_repetition(text):
    assert streaming_clean_text(text) == pipeline_clean_text(text)


@pytest.mark.parametrize("text", PLAIN_CASES)
def test_streaming_leaves_non_repetitive_text_alone(text):
    assert streaming_clean_text(text) == pipeline_clean_text(text)
