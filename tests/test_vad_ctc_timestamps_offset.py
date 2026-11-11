"""Fun-ASR-Nano ctc_timestamps must get the same VAD offset as timestamps."""

from unittest.mock import MagicMock, patch

import numpy as np

VAD_SEGMENTS = [[0, 4000], [5000, 9000]]


def _token(token, start, end):
    return {"token": token, "start_time": start, "end_time": end, "score": 1.0}


def _segment_result(index):
    # Segment-relative times in seconds, as FunASRNano.inference_llm returns them.
    return {
        "text": f"word{index}",
        "ctc_text": f"word{index}",
        "timestamps": [_token(f"word{index}", 0.12, 0.48)],
        "ctc_timestamps": [_token(f"word{index}", 0.12, 0.48)],
    }


@patch("funasr.auto.auto_model.slice_padding_audio_samples")
@patch("funasr.auto.auto_model.load_audio_text_image_video")
@patch("funasr.auto.auto_model.prepare_data_iterator")
def test_vad_offset_is_applied_to_ctc_timestamps(mock_prep, mock_load, mock_slice):
    from funasr.auto.auto_model import AutoModel

    am = AutoModel.__new__(AutoModel)
    am.model = MagicMock()
    am.vad_model = MagicMock()
    am.punc_model = None
    am.punc_kwargs = {}
    am.spk_model = None
    am.spk_kwargs = {}
    am.cb_model = None
    am.spk_mode = None
    am.vad_kwargs = {}
    am.kwargs = {
        "batch_size_s": 300,
        "batch_size_threshold_s": 60,
        "device": "cpu",
        "disable_pbar": True,
        "frontend": MagicMock(fs=16000),
        "fs": 16000,
    }
    am._reset_runtime_configs = MagicMock()

    def mock_inference(data, input_len=None, model=None, kwargs=None, **cfg):
        if model is am.vad_model:
            return [{"key": "utt", "value": [list(s) for s in VAD_SEGMENTS]}]
        return [_segment_result(int(sample[0])) for sample in data]

    am.inference = MagicMock(side_effect=mock_inference)
    mock_prep.return_value = (["utt"], [np.zeros(16000 * 9, dtype=np.float32)])
    mock_load.return_value = np.zeros(16000 * 9, dtype=np.float32)
    # Tag each sliced segment with its VAD index so the fake model knows which one it got.
    mock_slice.side_effect = lambda speech, speech_lengths, segments: (
        [np.full(16, VAD_SEGMENTS.index(seg), dtype=np.float32) for seg, _ in segments],
        [16 for _ in segments],
    )

    result = am.inference_with_vad("dummy_input")[0]

    spans = [(t["start_time"], t["end_time"]) for t in result["timestamps"]]
    assert spans == [(0.12, 0.48), (5.12, 5.48)]
    ctc_spans = [(t["start_time"], t["end_time"]) for t in result["ctc_timestamps"]]
    assert ctc_spans == spans
