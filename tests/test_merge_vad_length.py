"""Regression tests for the merge_length_s option of AutoModel.generate()."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np


VAD_SEGMENTS = [[0, 4000], [5000, 9000], [10000, 14000]]


class TestMergeVadLength(unittest.TestCase):
    def _make_auto_model(self, **init_kwargs):
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
            **init_kwargs,
        }
        am._reset_runtime_configs = MagicMock()

        def mock_inference(data, input_len=None, model=None, kwargs=None, **cfg):
            if model is am.vad_model:
                return [{"key": "utt", "value": [list(s) for s in VAD_SEGMENTS]}]
            return [{"text": "x"} for _ in data]

        am.inference = MagicMock(side_effect=mock_inference)
        return am

    @patch("funasr.auto.auto_model.slice_padding_audio_samples")
    @patch("funasr.auto.auto_model.load_audio_text_image_video")
    @patch("funasr.auto.auto_model.prepare_data_iterator")
    def _asr_segments(self, am, mock_prep, mock_load, mock_slice, **cfg):
        mock_prep.return_value = (["utt"], [np.zeros(16000 * 14, dtype=np.float32)])
        mock_load.return_value = np.zeros(16000 * 14, dtype=np.float32)
        mock_slice.side_effect = lambda speech, speech_lengths, segments: (
            [np.zeros(16000, dtype=np.float32) for _ in segments],
            [16000 for _ in segments],
        )

        am.inference_with_vad("dummy_input", merge_vad=True, **cfg)

        segments = []
        for call in mock_slice.call_args_list:
            segments.extend(seg for seg, _ in call.args[2])
        return sorted(segments)

    def test_default_merge_length_is_15_seconds(self):
        am = self._make_auto_model()
        self.assertEqual(self._asr_segments(am), [[0, 14000]])

    def test_generate_merge_length_s_is_applied(self):
        am = self._make_auto_model()
        self.assertEqual(self._asr_segments(am, merge_length_s=10), [[0, 9000], [9000, 14000]])

    def test_construction_merge_length_s_is_applied(self):
        am = self._make_auto_model(merge_length_s=10)
        self.assertEqual(self._asr_segments(am), [[0, 9000], [9000, 14000]])

    def test_generate_merge_length_s_overrides_construction_value(self):
        am = self._make_auto_model(merge_length_s=10)
        self.assertEqual(self._asr_segments(am, merge_length_s=15), [[0, 14000]])


if __name__ == "__main__":
    unittest.main()
