"""Byte-fragment spoken tokens must never be treated as punctuation.

Review regression for FunASR PR #3703 (issue #3702 follow-up): Nano's CTC
tokenizer is byte-based, so one spoken character can span several token ids
(official SenseVoiceTokenizer: 郗 -> [10958, 245]). All three Nano timestamp
paths decode each token id independently, and each fragment decodes to U+FFFD
while the id sequence decodes to real speech. The punctuation classifier must
not collapse such fragments' valid acoustic spans.

These tests instantiate the repository's real ``SenseVoiceTokenizer``
production path (``SenseVoiceTokenizer`` factory -> ``get_tokenizer`` ->
``get_encoding`` -> production ``Tokenizer``) against a deterministic
temporary offline ``.tiktoken`` vocab, so they run without the Fun-ASR-Nano
``multilingual.tiktoken`` model artifact (which ships with the model, not
this repo) and without weights/GPU. The fixture vocab mirrors the official
vocab shape: ordinary CJK chars and punctuation are single ids, while 郗
(deliberately given no merged rank) splits into single-byte ids whose
independent decodes are U+FFFD — the exact byte-fragment mechanism
(official: 郗 -> [10958, 245]). Official-artifact cross-check
(SenseVoiceTokenizer from FunAudioLLM/Fun-ASR-Nano-2512
``multilingual.tiktoken`` @272c57b, sha256 74797963...): 郗 -> [10958, 245],
per-id decodes U+FFFD, roundtrip intact; helper/vLLM/pipeline probes preserve
[0.60, 0.72]/[0.84, 0.96]-style spans post-fix (see PR review evidence).
"""

import base64
import types

import numpy as np
import pytest
import torch

from funasr.models.fun_asr_nano.tools.utils import (
    _classify_timestamp_token,
    anchor_punctuation_timestamps,
    forced_align,
)
from funasr.tokenizer.whisper_tokenizer import SenseVoiceTokenizer

TEXT = "我叫郗明。你好。"
FRAMES_PER_TOKEN = 10
FRAME_TO_SEC = 6 * 10 / 1000


def _write_fragment_vocab(path):
    """Deterministic offline vocab: byte-level base plus single-id merged
    ranks for every fixture char except 郗, which stays byte-split. Format
    matches production ``.tiktoken`` files (``base64(token) rank`` lines)."""
    lines = []
    for byte in range(256):
        lines.append(f"{base64.b64encode(bytes([byte])).decode()} {byte}")
    next_rank = 256
    # Prefix pair first so BPE completes each 3-byte merge (tiktoken merges
    # lowest-rank pairs first); 郗 gets no ranks and stays byte-split.
    for char in "我叫明你好。，":
        raw = char.encode("utf-8")
        assert len(raw) == 3, char
        lines.append(f"{base64.b64encode(raw[:2]).decode()} {next_rank}")
        next_rank += 1
        lines.append(f"{base64.b64encode(raw).decode()} {next_rank}")
        next_rank += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def sensevoice_tokenizer(tmp_path):
    """The real production Tokenizer via the SenseVoiceTokenizer factory."""
    vocab = tmp_path / "byte-fragment.tiktoken"
    _write_fragment_vocab(vocab)
    tokenizer = SenseVoiceTokenizer(vocab_path=str(vocab))
    target_ids = tokenizer.encode(TEXT)
    assert tokenizer.decode(target_ids) == TEXT
    return tokenizer


def _peaky_log_probs(target_ids, vocab_size, frames_per_token=FRAMES_PER_TOKEN):
    n_frames = len(target_ids) * frames_per_token
    log_probs = torch.full((n_frames, vocab_size), -30.0)
    for i, tid in enumerate(target_ids):
        log_probs[i * frames_per_token : (i + 1) * frames_per_token, tid] = 0.0
    return log_probs


def _blank_id(tokenizer):
    return tokenizer.get_vocab_size() - 1


def _production_timestamps(tokenizer, target_ids):
    """Mirror the production decode-per-id + scale path shared by all three
    Nano timestamp methods, over the repo's real forced_align."""
    items = forced_align(
        _peaky_log_probs(target_ids, tokenizer.get_vocab_size()),
        torch.tensor(target_ids, dtype=torch.int64),
        _blank_id(tokenizer),
    )
    assert [it["token"] for it in items] == list(target_ids)
    return [
        {
            "token": tokenizer.decode([it["token"]]),
            "start_time": it["start_time"] * FRAME_TO_SEC,
            "end_time": it["end_time"] * FRAME_TO_SEC,
        }
        for it in items
    ]


def test_undecodable_fragment_is_spoken_not_punctuation():
    assert _classify_timestamp_token("�") == "spoken"
    assert _classify_timestamp_token("��") == "spoken"
    # Genuine punctuation still classifies as punctuation.
    assert _classify_timestamp_token("。") == "punctuation"
    assert _classify_timestamp_token(",") == "punctuation"
    # Special tokens keep their own class.
    assert _classify_timestamp_token("<sil>") == "special"


def test_byte_fragment_spans_survive_anchoring(sensevoice_tokenizer):
    tokenizer = sensevoice_tokenizer
    target_ids = tokenizer.encode(TEXT)
    assert len(target_ids) > len(TEXT), "郗 must stay byte-split"
    assert tokenizer.decode(target_ids) == TEXT
    assert any(tokenizer.decode([i]) == "�" for i in target_ids)

    timestamps = _production_timestamps(tokenizer, target_ids)
    before = [(t["token"], t["start_time"], t["end_time"]) for t in timestamps]
    anchor_punctuation_timestamps(timestamps)

    for (token, start, end), ts in zip(before, timestamps):
        if token == "�":
            assert (ts["start_time"], ts["end_time"]) == (start, end)
            assert ts["end_time"] > ts["start_time"], "fragment keeps acoustic extent"
    # Genuine 。 between speech still anchors (control).
    punct = [t for t in timestamps if t["token"] == "。"]
    assert punct, "expected genuine punctuation in fixtures"
    assert punct[0]["start_time"] == punct[0]["end_time"]


def test_vllm_method_preserves_byte_fragment_spans(sensevoice_tokenizer):
    from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

    tokenizer = sensevoice_tokenizer
    target_ids = tokenizer.encode(TEXT)
    n_frames = len(target_ids) * FRAMES_PER_TOKEN
    log_probs = _peaky_log_probs(target_ids, tokenizer.get_vocab_size())

    engine = FunASRNanoVLLM.__new__(FunASRNanoVLLM)
    engine.ctc_decoder = lambda e, l: (log_probs.unsqueeze(0), torch.tensor([n_frames]))
    engine.ctc = types.SimpleNamespace(log_softmax=lambda d: d)
    engine.ctc_tokenizer = tokenizer
    engine.blank_id = _blank_id(tokenizer)

    result = engine._compute_timestamps(
        torch.zeros(1, n_frames, 8), torch.tensor([n_frames]), TEXT
    )
    fragments = [t for t in result if t["token"] == "�"]
    assert len(fragments) >= 2
    for t in fragments:
        assert t["end_time"] > t["start_time"]
    assert any(t["token"] == "。" and t["start_time"] == t["end_time"] for t in result)


def test_pipeline_method_preserves_byte_fragment_spans(
    sensevoice_tokenizer, monkeypatch
):
    from funasr.models.fun_asr_nano.inference_vllm_pipeline import (
        FunASRNanoVLLMPipeline,
    )

    tokenizer = sensevoice_tokenizer
    target_ids = tokenizer.encode(TEXT)
    n_frames = len(target_ids) * FRAMES_PER_TOKEN
    log_probs = _peaky_log_probs(target_ids, tokenizer.get_vocab_size())

    monkeypatch.setattr(
        "funasr.utils.load_utils.extract_fbank",
        lambda *a, **k: (torch.zeros(1, 4, 80), torch.tensor([[4]])),
    )
    pipeline = FunASRNanoVLLMPipeline.__new__(FunASRNanoVLLMPipeline)
    pipeline.device = "cpu"
    pipeline.asr_engine = types.SimpleNamespace(
        frontend=object(),
        audio_encoder=lambda s, sl: (
            torch.zeros(1, n_frames, 8),
            torch.tensor([n_frames]),
        ),
        ctc_decoder=lambda e, l: (log_probs.unsqueeze(0), torch.tensor([n_frames])),
        ctc=types.SimpleNamespace(log_softmax=lambda d: d),
        ctc_tokenizer=tokenizer,
        blank_id=_blank_id(tokenizer),
    )
    result = pipeline._compute_all_timestamps(
        [np.zeros(32000, dtype=np.float32)], [[1000, 9000]], [TEXT]
    )
    fragments = [t for t in result if t["token"] == "�"]
    assert len(fragments) >= 2
    for t in fragments:
        assert t["end_time"] > t["start_time"]
    # +1s VAD offset applied on top of preserved spans: compare against the
    # un-offset helper path for the identical target sequence.
    expected = _production_timestamps(tokenizer, target_ids)
    expected_frags = [t for t in expected if t["token"] == "�"]
    assert len(fragments) == len(expected_frags)
    for got, want in zip(fragments, expected_frags):
        assert got["start_time"] == pytest.approx(want["start_time"] + 1.0)
        assert got["end_time"] == pytest.approx(want["end_time"] + 1.0)
