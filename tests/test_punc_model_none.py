"""Tests for issue #2839: punc_model=None or empty string should not cause UnboundLocalError."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class TestPuncModelNone(unittest.TestCase):
    """Test that inference_with_vad works when punc_model is None."""

    def _make_auto_model(self, punc_model=None, spk_model=None, spk_mode=None):
        """Create a minimal AutoModel instance with mocked dependencies."""
        from funasr.auto.auto_model import AutoModel

        am = AutoModel.__new__(AutoModel)
        am.model = MagicMock()
        am.vad_model = MagicMock()
        am.punc_model = punc_model
        am.punc_kwargs = {}
        am.spk_model = spk_model
        am.cb_model = None
        am.spk_mode = spk_mode
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
        return am

    def _setup_mocks(self, am, mock_slice, mock_load, mock_prep):
        """Configure standard mocks for a single-segment VAD + ASR flow."""
        # VAD returns one segment [0, 16000ms]
        vad_result = [{"key": "test_utt", "value": [[0, 16000]]}]
        # ASR returns text with timestamps
        asr_result = [{"text": "hello world", "timestamp": [[0, 500], [500, 1000]]}]

        call_count = [0]
        results_seq = [vad_result, asr_result]

        def mock_inference(data, input_len=None, model=None, kwargs=None, **cfg):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(results_seq):
                return results_seq[idx]
            return [{"text": ""}]

        am.inference = MagicMock(side_effect=mock_inference)
        mock_prep.return_value = (["test_utt"], [np.zeros(16000, dtype=np.float32)])
        mock_load.return_value = np.zeros(16000, dtype=np.float32)
        mock_slice.return_value = ([np.zeros(16000, dtype=np.float32)], [16000])

    @patch("funasr.auto.auto_model.slice_padding_audio_samples")
    @patch("funasr.auto.auto_model.load_audio_text_image_video")
    @patch("funasr.auto.auto_model.prepare_data_iterator")
    def test_punc_model_none_basic(self, mock_prep, mock_load, mock_slice):
        """Basic inference with punc_model=None should not raise UnboundLocalError."""
        am = self._make_auto_model(punc_model=None)
        self._setup_mocks(am, mock_slice, mock_load, mock_prep)

        results = am.inference_with_vad("dummy_input")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "hello world")
        self.assertEqual(results[0]["key"], "test_utt")

    @patch("funasr.auto.auto_model.slice_padding_audio_samples")
    @patch("funasr.auto.auto_model.load_audio_text_image_video")
    @patch("funasr.auto.auto_model.prepare_data_iterator")
    def test_sentence_timestamp_with_punc_model_none(self, mock_prep, mock_load, mock_slice):
        """sentence_timestamp=True with punc_model=None should not crash."""
        am = self._make_auto_model(punc_model=None)
        self._setup_mocks(am, mock_slice, mock_load, mock_prep)

        # This path previously caused UnboundLocalError on punc_res
        results = am.inference_with_vad("dummy_input", sentence_timestamp=True)

        self.assertEqual(len(results), 1)
        # sentence_info should be empty list since punc_res is unavailable
        self.assertEqual(results[0].get("sentence_info"), [])

    @patch("funasr.auto.auto_model.slice_padding_audio_samples")
    @patch("funasr.auto.auto_model.load_audio_text_image_video")
    @patch("funasr.auto.auto_model.prepare_data_iterator")
    def test_punc_model_with_value_still_works(self, mock_prep, mock_load, mock_slice):
        """When punc_model is provided, punc_res should still be used normally."""
        punc_mock = MagicMock()
        am = self._make_auto_model(punc_model=punc_mock)

        vad_result = [{"key": "test_utt", "value": [[0, 16000]]}]
        asr_result = [{"text": "hello world", "timestamp": [[0, 500], [500, 1000]]}]
        punc_result = [{"text": "Hello, world.", "punc_array": [1, 2]}]

        call_count = [0]
        results_seq = [vad_result, asr_result, punc_result]

        def mock_inference(data, input_len=None, model=None, kwargs=None, **cfg):
            idx = call_count[0]
            call_count[0] += 1
            return results_seq[idx]

        am.inference = MagicMock(side_effect=mock_inference)
        mock_prep.return_value = (["test_utt"], [np.zeros(16000, dtype=np.float32)])
        mock_load.return_value = np.zeros(16000, dtype=np.float32)
        mock_slice.return_value = ([np.zeros(16000, dtype=np.float32)], [16000])

        results = am.inference_with_vad("dummy_input")

        self.assertEqual(len(results), 1)
        # Text should be updated with punctuated version
        self.assertEqual(results[0]["text"], "Hello, world.")


if __name__ == "__main__":
    unittest.main()
