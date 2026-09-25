"""Semantic-symbol tokens must keep their acoustic spans (PR #3703 rework).

LauraGPT's blocking review on PR #3703 (comment 3996970511): the accepted
U+FFFD byte-fragment guard fixed undecodable fragments, but the classifier
still treated every non-word character as punctuation. With the official
SenseVoice vocabulary, ``我用C++。你好。`` encodes ``++`` as one token (id
24754, Unicode Sm) and ``价格是$10。你好。`` encodes ``$`` as one token (id 3,
Sc). Both decode without replacement characters, are spoken semantic
content, and had their aligned spans [1.8, 2.4] collapsed to [1.8, 1.8] by
the actual ``FunASRNanoVLLM._compute_timestamps`` (the pipeline path after
its +1s VAD offset: [2.8, 3.4] -> [2.8, 2.8]).

The classifier now anchors only the supported sentence-punctuation set
(``_SENTENCE_PUNCTUATION_CHARS``); everything else defaults to spoken.

These tests instantiate the repository's real ``SenseVoiceTokenizer``
production path (``SenseVoiceTokenizer`` factory -> ``get_tokenizer`` ->
``get_encoding`` -> production ``Tokenizer``) against a deterministic
temporary offline ``.tiktoken`` vocab, so they run without the Fun-ASR-Nano
``multilingual.tiktoken`` model artifact (which ships with the model, not
this repo) and without weights/GPU. The fixture vocab mirrors the official
vocab shape for both reviewer cases: CJK chars and ``。`` are single ids,
``++`` is one merged id (official: id 24754), ``$`` decodes standalone
(official: id 3, a single-byte rank), ``C`` stays a single-byte id
(official: id 34), and ``10`` is one merged id (official: id 3254).
Official-artifact cross-check (SenseVoiceTokenizer from
FunAudioLLM/Fun-ASR-Nano-2512 ``multilingual.tiktoken`` @272c57b, sha256
74797963e813193436aabcff7c1c235d37de8097b71c563ec8b63b7a515c718):
``我用C++。你好。`` -> [51405, 53719, 34, 24754, 1542, 48934, 50371, 1542]
and ``价格是$10。你好。`` -> [48869, 52258, 51971, 3, 3254, 1542, 48934,
50371, 1542]; the symbol ids decode standalone to ``++`` / ``$`` while
``。`` is id 1542.
"""

import base64
import types

import numpy as np
import pytest
import torch

from funasr.models.fun_asr_nano.tools.utils import (
    _classify_timestamp_token,
    _is_sentence_punctuation,
    anchor_punctuation_timestamps,
    forced_align,
)
from funasr.tokenizer.whisper_tokenizer import SenseVoiceTokenizer

TEXT_PLUS = "我用C++。你好。"
TEXT_CURRENCY = "价格是$10。你好。"
FRAMES_PER_TOKEN = 10
FRAME_TO_SEC = 6 * 10 / 1000

# Representative official-vocab symbol tokens (FunAudioLLM/Fun-ASR-Nano-2512
# @272c57b): all are single-id spoken semantic content, never sentence
# punctuation. Categories are Unicode general categories.
OFFICIAL_SYMBOL_TOKENS = [
    ("++", "Sm"),
    ("+", "Sm"),
    ("=", "Sm"),
    ("<", "Sm"),
    (">", "Sm"),
    ("^", "Sk"),
    ("~", "Sm"),
    ("$", "Sc"),
    ("€", "Sc"),
    ("℃", "So"),
    ("°", "So"),
    ("%", "Po"),
    ("#", "Po"),
    ("@", "Po"),
    ("&", "Po"),
    ("*", "Po"),
    ("/", "Po"),
    ("-", "Pd"),
    ("—", "Pd"),
    ("…", "Po"),
    ('"', "Pe"),
    ("(", "Ps"),
    ("「", "Ps"),
    ("》", "Pe"),
]

# Sentence punctuation the anchoring policy supports: terminators and clause
# separators that ASR emits as structural markup with no acoustic extent.
SUPPORTED_SENTENCE_PUNCTUATION = [
    "。",
    "，",
    "、",
    "；",
    "：",
    "！",
    "？",
    ".",
    ",",
    "!",
    "?",
    ";",
    ":",
]


def _write_semantic_symbol_vocab(path):
    """Deterministic offline vocab: byte-level base plus merged ranks that
    mirror the official reviewer-case tokenization. Format matches production
    ``.tiktoken`` files (``base64(token) rank`` lines)."""
    lines = []
    for byte in range(256):
        lines.append(f"{base64.b64encode(bytes([byte])).decode()} {byte}")
    next_rank = 256
    # CJK chars and 。/，: prefix pair first so BPE completes each 3-byte
    # merge (tiktoken merges lowest-rank pairs first).
    for char in "我用你好价格是。，":
        raw = char.encode("utf-8")
        assert len(raw) == 3, char
        lines.append(f"{base64.b64encode(raw[:2]).decode()} {next_rank}")
        next_rank += 1
        lines.append(f"{base64.b64encode(raw).decode()} {next_rank}")
        next_rank += 1
    # Multi-char symbol tokens, mirroring official ids 24754 ("++") and 3254
    # ("10"): single merged ranks, so each decodes standalone as one token.
    for token in ("++", "10"):
        lines.append(f"{base64.b64encode(token.encode('utf-8')).decode()} {next_rank}")
        next_rank += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def sensevoice_tokenizer(tmp_path):
    """The real production Tokenizer via the SenseVoiceTokenizer factory."""
    vocab = tmp_path / "semantic-symbol.tiktoken"
    _write_semantic_symbol_vocab(vocab)
    tokenizer = SenseVoiceTokenizer(vocab_path=str(vocab))
    for text in (TEXT_PLUS, TEXT_CURRENCY):
        assert tokenizer.decode(tokenizer.encode(text)) == text
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


def test_supported_sentence_punctuation_anchors():
    for token in SUPPORTED_SENTENCE_PUNCTUATION:
        assert _is_sentence_punctuation(token), token
        assert _classify_timestamp_token(token) == "punctuation", token
    # Combinations of supported punctuation stay punctuation.
    assert _is_sentence_punctuation("!?")
    assert _is_sentence_punctuation("!!")
    assert _classify_timestamp_token("!?") == "punctuation"


def test_semantic_symbols_are_spoken_not_punctuation():
    for token, _category in OFFICIAL_SYMBOL_TOKENS:
        assert _classify_timestamp_token(token) == "spoken", token
        assert not _is_sentence_punctuation(token), token
    # A symbol glued to sentence punctuation must not anchor either.
    assert _classify_timestamp_token('。"') == "spoken"
    assert _classify_timestamp_token("%,") == "spoken"
    # Undecodable byte fragments stay spoken; special tokens keep their class.
    assert _classify_timestamp_token("�") == "spoken"
    assert _classify_timestamp_token("<sil>") == "special"


def test_symbol_token_ids_decode_standalone(sensevoice_tokenizer):
    tokenizer = sensevoice_tokenizer
    plus_ids = tokenizer.encode(TEXT_PLUS)
    currency_ids = tokenizer.encode(TEXT_CURRENCY)
    assert tokenizer.decode(plus_ids) == TEXT_PLUS
    assert tokenizer.decode(currency_ids) == TEXT_CURRENCY
    # ++ and $ must be single ids that decode standalone (official: one id
    # each, 24754 and 3), not per-character or byte-split.
    plus_tokens = [tokenizer.decode([i]) for i in plus_ids]
    currency_tokens = [tokenizer.decode([i]) for i in currency_ids]
    assert "++" in plus_tokens, plus_tokens
    assert "$" in currency_tokens, currency_tokens
    assert _classify_timestamp_token("++") == "spoken"
    assert _classify_timestamp_token("$") == "spoken"


def test_semantic_symbol_spans_survive_anchoring(sensevoice_tokenizer):
    tokenizer = sensevoice_tokenizer
    for text, symbol in ((TEXT_PLUS, "++"), (TEXT_CURRENCY, "$")):
        target_ids = tokenizer.encode(text)
        timestamps = _production_timestamps(tokenizer, target_ids)
        before = [(t["token"], t["start_time"], t["end_time"]) for t in timestamps]
        anchor_punctuation_timestamps(timestamps)

        symbols = [t for t in timestamps if t["token"] == symbol]
        assert symbols, f"expected {symbol!r} in {text!r}"
        for (token, start, end), ts in zip(before, timestamps):
            if token == symbol:
                # The acoustic span is preserved verbatim: no collapse.
                assert (ts["start_time"], ts["end_time"]) == (start, end)
                assert ts["end_time"] > ts["start_time"], f"{symbol} keeps acoustic extent"
        # Genuine 。 between speech still anchors (zero-width control): it
        # pins at the preceding spoken token's end, which for these texts is
        # the symbol or the token right after it.
        punct = [t for t in timestamps if t["token"] == "。"]
        assert punct
        assert punct[0]["start_time"] == punct[0]["end_time"]
        punct_index = next(i for i, t in enumerate(before) if t[0] == "。")
        assert before[punct_index - 1][2] == punct[0]["start_time"]


def test_vllm_method_preserves_semantic_symbol_spans(sensevoice_tokenizer):
    from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

    tokenizer = sensevoice_tokenizer
    for text, symbol in ((TEXT_PLUS, "++"), (TEXT_CURRENCY, "$")):
        target_ids = tokenizer.encode(text)
        n_frames = len(target_ids) * FRAMES_PER_TOKEN
        log_probs = _peaky_log_probs(target_ids, tokenizer.get_vocab_size())

        engine = FunASRNanoVLLM.__new__(FunASRNanoVLLM)
        engine.ctc_decoder = lambda e, l, lp=log_probs, nf=n_frames: (
            lp.unsqueeze(0),
            torch.tensor([nf]),
        )
        engine.ctc = types.SimpleNamespace(log_softmax=lambda d: d)
        engine.ctc_tokenizer = tokenizer
        engine.blank_id = _blank_id(tokenizer)

        result = engine._compute_timestamps(
            torch.zeros(1, n_frames, 8), torch.tensor([n_frames]), text
        )
        symbols = [t for t in result if t["token"] == symbol]
        assert symbols
        for t in symbols:
            assert t["end_time"] > t["start_time"], f"{symbol} span preserved"
        # Genuine punctuation still anchors between speech.
        assert any(t["token"] == "。" and t["start_time"] == t["end_time"] for t in result)


def test_pipeline_method_preserves_semantic_symbol_spans(sensevoice_tokenizer, monkeypatch):
    from funasr.models.fun_asr_nano.inference_vllm_pipeline import (
        FunASRNanoVLLMPipeline,
    )

    tokenizer = sensevoice_tokenizer
    for text, symbol in ((TEXT_PLUS, "++"), (TEXT_CURRENCY, "$")):
        target_ids = tokenizer.encode(text)
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
            audio_encoder=lambda s, sl, nf=n_frames: (
                torch.zeros(1, nf, 8),
                torch.tensor([nf]),
            ),
            ctc_decoder=lambda e, l, lp=log_probs, nf=n_frames: (
                lp.unsqueeze(0),
                torch.tensor([nf]),
            ),
            ctc=types.SimpleNamespace(log_softmax=lambda d: d),
            ctc_tokenizer=tokenizer,
            blank_id=_blank_id(tokenizer),
        )
        result = pipeline._compute_all_timestamps(
            [np.zeros(32000, dtype=np.float32)], [[1000, 9000]], [text]
        )
        symbols = [t for t in result if t["token"] == symbol]
        assert symbols
        # +1s VAD offset applied on top of preserved spans: compare against
        # the un-offset helper path for the identical target sequence.
        expected = _production_timestamps(tokenizer, target_ids)
        expected_symbols = [t for t in expected if t["token"] == symbol]
        assert len(symbols) == len(expected_symbols)
        for got, want in zip(symbols, expected_symbols):
            assert got["start_time"] == pytest.approx(want["start_time"] + 1.0)
            assert got["end_time"] == pytest.approx(want["end_time"] + 1.0)
            assert got["end_time"] > got["start_time"], f"{symbol} span preserved"
        # Genuine punctuation still anchors between speech.
        assert any(t["token"] == "。" and t["start_time"] == t["end_time"] for t in result)
