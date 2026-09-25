"""Punctuation-timestamp anchoring for Fun-ASR-Nano forced alignment.

Issue #3702: with merge_vad, a sentence-final punctuation token (which has no
acoustic realization) is placed by CTC forced alignment at the next sentence's
onset frame, so the current sentence's punctuation timestamp lands on the next
sentence start. Proven with synthetic peaky emissions through the repo's exact
``forced_align`` (see run evidence): in a merged [sent1 + silence + sent2]
window the punctuation lands at the silence end, while the single-sentence
control keeps it at the window end.

These tests pin the post-processing rule: a punctuation-only token that is
followed by more speech is anchored (zero-width) at the preceding spoken
token's end. Punctuation carries no acoustic extent, so anchoring invents no
duration. Leading punctuation (no predecessor) and trailing punctuation (no
successor, e.g. single-segment output) are left untouched, preserving
existing single-segment behavior.
"""

import copy

import pytest

from funasr.models.fun_asr_nano.tools.utils import anchor_punctuation_timestamps


def _ts(token, start, end):
    return {"token": token, "start_time": start, "end_time": end}


def test_punctuation_anchored_at_previous_spoken_end():
    # Mirrors the proven repro: 好 ends at 19, 。misplaced at [72, 73).
    timestamps = [_ts("好", 18, 19), _ts("。", 72, 73), _ts("他", 78, 79)]
    result = anchor_punctuation_timestamps(timestamps)
    assert result[1] == {"token": "。", "start_time": 19, "end_time": 19}
    assert result[0] == {"token": "好", "start_time": 18, "end_time": 19}
    assert result[2] == {"token": "他", "start_time": 78, "end_time": 79}


def test_leading_punctuation_without_predecessor_is_kept():
    timestamps = [_ts("。", 5, 6), _ts("好", 18, 19)]
    before = copy.deepcopy(timestamps)
    assert anchor_punctuation_timestamps(timestamps) == before


def test_consecutive_punctuation_share_one_anchor():
    timestamps = [_ts("好", 18, 19), _ts("，", 50, 51), _ts("。", 72, 73), _ts("他", 78, 79)]
    result = anchor_punctuation_timestamps(timestamps)
    assert (result[1]["start_time"], result[1]["end_time"]) == (19, 19)
    assert (result[2]["start_time"], result[2]["end_time"]) == (19, 19)
    assert (result[3]["start_time"], result[3]["end_time"]) == (78, 79)


def test_spoken_tokens_are_never_rewritten():
    timestamps = [_ts("你", 12, 13), _ts("好", 18, 19), _ts("。", 39, 40)]
    before = copy.deepcopy(timestamps)
    result = anchor_punctuation_timestamps(timestamps)
    assert result[0] == before[0]
    # Single-sentence control shape: punctuation already at window end is kept.
    assert result[2] == before[2]


def test_special_tokens_are_not_treated_as_punctuation():
    timestamps = [_ts("<sil>", 0, 5), _ts("好", 18, 19)]
    before = copy.deepcopy(timestamps)
    assert anchor_punctuation_timestamps(timestamps) == before


def test_punctuation_followed_only_by_special_is_kept():
    # No later real speech: the trailing <sil> must not make 。 eligible.
    timestamps = [_ts("好", 18, 19), _ts("。", 72, 73), _ts("<sil>", 80, 85)]
    before = copy.deepcopy(timestamps)
    assert anchor_punctuation_timestamps(timestamps) == before


def test_special_token_is_not_used_as_predecessor_anchor():
    # The <sil> between speech and punctuation must not become the anchor.
    timestamps = [_ts("好", 18, 19), _ts("<sil>", 30, 35), _ts("。", 72, 73), _ts("他", 78, 79)]
    result = anchor_punctuation_timestamps(timestamps)
    assert (result[2]["start_time"], result[2]["end_time"]) == (19, 19)
    assert (result[1]["start_time"], result[1]["end_time"]) == (30, 35)
    assert (result[3]["start_time"], result[3]["end_time"]) == (78, 79)


def test_pipeline_segment_local_anchor_boundary(monkeypatch):
    """Pipeline segment boundary: anchoring applies per VAD segment, never
    across segments. segA shows intra-segment anchoring (misplaced 。 before
    later speech in the SAME aligned segment anchors); segB shows a
    segment-final 。 staying put even though segC contains later speech;
    VAD offsets and spoken spans pass through verbatim."""
    import types

    import numpy as np
    import torch

    from funasr.models.fun_asr_nano.inference_vllm_pipeline import FunASRNanoVLLMPipeline

    id2char = {2: "好", 3: "。", 4: "他", 5: "在", 6: "是"}
    spans = iter(
        [
            [
                {"token": 2, "start_time": 18, "end_time": 19},
                {"token": 3, "start_time": 72, "end_time": 73},
                {"token": 4, "start_time": 78, "end_time": 79},
            ],
            [
                {"token": 6, "start_time": 10, "end_time": 12},
                {"token": 3, "start_time": 40, "end_time": 41},
            ],
            [{"token": 5, "start_time": 50, "end_time": 56}],
        ]
    )

    class StubTok:
        def encode(self, text):
            return [1] * max(1, len(text))

        def decode(self, ids):
            return "".join(id2char.get(i, "?") for i in ids)

    engine = types.SimpleNamespace(
        frontend=object(),
        device="cpu",
        blank_id=0,
        audio_encoder=lambda s, sl: (torch.zeros(1, 10, 8), torch.tensor([10])),
        ctc_decoder=lambda e, l: (torch.zeros(1, 10, 8), torch.tensor([10])),
        ctc=types.SimpleNamespace(log_softmax=lambda d: d),
        ctc_tokenizer=StubTok(),
    )
    monkeypatch.setattr(
        "funasr.utils.load_utils.extract_fbank",
        lambda *a, **k: (torch.zeros(1, 4, 80), torch.tensor([[4]])),
    )
    monkeypatch.setattr(
        "funasr.models.fun_asr_nano.tools.utils.forced_align",
        lambda *a, **k: next(spans),
    )

    pipeline = FunASRNanoVLLMPipeline.__new__(FunASRNanoVLLMPipeline)
    pipeline.asr_engine = engine
    pipeline.device = "cpu"
    result = pipeline._compute_all_timestamps(
        [
            np.zeros(32000, dtype=np.float32),
            np.zeros(16000, dtype=np.float32),
            np.zeros(16000, dtype=np.float32),
        ],
        [[1000, 3000], [5000, 6000], [9000, 10000]],
        ["segAtext", "segBtext", "segCtext"],
    )

    assert [t["token"] for t in result] == ["好", "。", "他", "是", "。", "在"]
    # segA (offset 1.0s): intra-segment 。 anchors at 好.end bit-identically.
    assert result[1]["start_time"] == result[0]["end_time"]
    assert result[1]["end_time"] == result[0]["end_time"]
    assert result[0]["end_time"] == pytest.approx(2.14)
    assert result[2]["start_time"] == pytest.approx(5.68)
    # segB (offset 5.0s): segment-final 。 keeps its aligned span even though
    # segC contains later speech — cross-segment speech must not trigger it.
    assert result[4]["start_time"] == pytest.approx(7.4)
    assert result[4]["end_time"] == pytest.approx(7.46)
    assert result[3]["end_time"] == pytest.approx(5.72)
    # segC (offset 9.0s) passes through untouched.
    assert result[5]["start_time"] == pytest.approx(12.0)
    assert result[5]["end_time"] == pytest.approx(12.36)
