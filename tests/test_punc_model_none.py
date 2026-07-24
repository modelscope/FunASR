"""Regression tests for AutoModel VAD punctuation and sentence timestamps."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch


class TestPuncModelNone(unittest.TestCase):
    """Test that inference_with_vad works when punc_model is None."""

    def _make_auto_model(self, punc_model=None, spk_model=None, spk_mode=None):
        """Create a minimal AutoModel instance with mocked dependencies."""
        from funasr.auto.auto_model import AutoModel

        am = AutoModel.__new__(AutoModel)
        am.model = MagicMock()
        am.vad_model = MagicMock()
        am.punc_model = punc_model
        if punc_model is not None:
            punc_model.jieba_usr_dict = None
            punc_model.punc_list = ["<unk>", "_", "，", "。", "？", "、"]
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

    @patch("funasr.auto.auto_model.slice_padding_audio_samples")
    @patch("funasr.auto.auto_model.load_audio_text_image_video")
    @patch("funasr.auto.auto_model.prepare_data_iterator")
    def test_sentence_timestamp_uses_asr_words_for_unspaced_text(
        self, mock_prep, mock_load, mock_slice
    ):
        """Sentence timestamps must align SenseVoice words with token timestamps."""
        am = self._make_auto_model(punc_model=MagicMock())
        vad_result = [{"key": "test_utt", "value": [[0, 2000]]}]
        asr_result = [
            {
                "text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好世界",
                "timestamp": [[0, 500], [500, 1000], [1000, 1500], [1500, 2000]],
                "words": ["你", "好", "世", "界"],
            }
        ]
        punc_result = [{"text": "你好，世界。", "punc_array": [1, 2, 1, 3]}]
        results_seq = [vad_result, asr_result, punc_result]

        def mock_inference(data, *args, **kwargs):
            if len(results_seq) == 1:
                self.assertEqual(data, "你好世界")
            return results_seq.pop(0)

        am.inference = MagicMock(side_effect=mock_inference)
        mock_prep.return_value = (["test_utt"], [np.zeros(2000, dtype=np.float32)])
        mock_load.return_value = np.zeros(32000, dtype=np.float32)
        mock_slice.return_value = ([np.zeros(32000, dtype=np.float32)], [32000])

        results = am.inference_with_vad(
            "dummy_input", sentence_timestamp=True, return_raw_text=True
        )

        self.assertEqual(
            results[0]["raw_text"],
            "<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好世界",
        )
        self.assertEqual(
            results[0]["sentence_info"],
            [
                {
                    "text": "你好，",
                    "start": 0,
                    "end": 1000,
                    "timestamp": [[0, 500], [500, 1000]],
                    "raw_text": "你好",
                },
                {
                    "text": "世界。",
                    "start": 1000,
                    "end": 2000,
                    "timestamp": [[1000, 1500], [1500, 2000]],
                    "raw_text": "世界",
                },
            ],
        )

    @patch("funasr.auto.auto_model.slice_padding_audio_samples")
    @patch("funasr.auto.auto_model.load_audio_text_image_video")
    @patch("funasr.auto.auto_model.prepare_data_iterator")
    def test_sentence_timestamp_aligns_words_across_vad_segments(
        self, mock_prep, mock_load, mock_slice
    ):
        """Every SenseVoice VAD prefix must be excluded from punctuation alignment."""
        am = self._make_auto_model(punc_model=MagicMock())
        tag = "<|zh|><|NEUTRAL|><|Speech|><|woitn|>"
        results_seq = [
            [{"key": "test_utt", "value": [[0, 1000], [1000, 2000]]}],
            [{"text": f"{tag}你好", "timestamp": [[0, 500], [500, 1000]], "words": ["你", "好"]}],
            [{"text": f"{tag}世界", "timestamp": [[0, 500], [500, 1000]], "words": ["世", "界"]}],
            [{"text": "你好，世界。", "punc_array": [1, 2, 1, 3]}],
        ]

        def mock_inference(data, *args, **kwargs):
            if len(results_seq) == 1:
                self.assertEqual(data, "你好世界")
            return results_seq.pop(0)

        am.inference = MagicMock(side_effect=mock_inference)
        mock_prep.return_value = (["test_utt"], [np.zeros(32000, dtype=np.float32)])
        mock_load.return_value = np.zeros(32000, dtype=np.float32)
        mock_slice.return_value = ([np.zeros(16000, dtype=np.float32)], [16000])

        results = am.inference_with_vad("dummy_input", sentence_timestamp=True)

        self.assertEqual(
            [item["text"] for item in results[0]["sentence_info"]], ["你好，", "世界。"]
        )
        self.assertEqual(
            [item["timestamp"] for item in results[0]["sentence_info"]],
            [
                [[0, 500], [500, 1000]],
                [[1000, 1500], [1500, 2000]],
            ],
        )

    @patch("funasr.auto.auto_model.distribute_spk")
    @patch("funasr.auto.auto_model.postprocess")
    @patch("funasr.auto.auto_model.sv_chunk")
    @patch("funasr.auto.auto_model.slice_padding_audio_samples")
    @patch("funasr.auto.auto_model.load_audio_text_image_video")
    @patch("funasr.auto.auto_model.prepare_data_iterator")
    def test_speaker_punc_segment_uses_aligned_words(
        self,
        mock_prep,
        mock_load,
        mock_slice,
        mock_sv_chunk,
        mock_postprocess,
        mock_distribute_spk,
    ):
        """Speaker punctuation segmentation must use the same SenseVoice alignment."""
        am = self._make_auto_model(
            punc_model=MagicMock(), spk_model=MagicMock(), spk_mode="punc_segment"
        )
        am.cb_model = MagicMock(return_value=np.array([0]))
        tag = "<|zh|><|NEUTRAL|><|Speech|><|woitn|>"
        results_seq = [
            [{"key": "test_utt", "value": [[0, 2000]]}],
            [
                {
                    "text": f"{tag}你好世界",
                    "timestamp": [[0, 500], [500, 1000], [1000, 1500], [1500, 2000]],
                    "words": ["你", "好", "世", "界"],
                }
            ],
            [{"spk_embedding": torch.tensor([[1.0, 0.0]])}],
            [{"text": "你好，世界。", "punc_array": [1, 2, 1, 3]}],
        ]

        def mock_inference(data, *args, **kwargs):
            if len(results_seq) == 1:
                self.assertEqual(data, "你好世界")
            return results_seq.pop(0)

        am.inference = MagicMock(side_effect=mock_inference)
        mock_prep.return_value = (["test_utt"], [np.zeros(32000, dtype=np.float32)])
        mock_load.return_value = np.zeros(32000, dtype=np.float32)
        mock_slice.return_value = ([np.zeros(32000, dtype=np.float32)], [32000])
        mock_sv_chunk.return_value = [[0.0, 2.0, np.zeros(32000, dtype=np.float32)]]
        mock_postprocess.return_value = [{"start": 0.0, "end": 2.0, "spk": 0}]

        results = am.inference_with_vad("dummy_input")

        self.assertEqual(
            [item["text"] for item in results[0]["sentence_info"]], ["你好，", "世界。"]
        )
        mock_distribute_spk.assert_called_once()

    @patch("funasr.auto.auto_model.slice_padding_audio_samples")
    @patch("funasr.auto.auto_model.load_audio_text_image_video")
    @patch("funasr.auto.auto_model.prepare_data_iterator")
    def test_punctuation_preserves_english_surface_text(self, mock_prep, mock_load, mock_slice):
        """Timestamp/BPE units must not add spaces to contractions, URLs, or emails."""
        am = self._make_auto_model(punc_model=MagicMock())
        tag = "<|en|><|NEUTRAL|><|Speech|><|woitn|>"
        surface_text = "don't stop https://nature.com email@example.com"
        words = [
            "don",
            "'",
            "t",
            "stop",
            "https",
            ":",
            "/",
            "/",
            "nature",
            ".",
            "com",
            "email",
            "@",
            "example",
            ".",
            "com",
        ]
        mock_prep.return_value = (["test_utt"], [np.zeros(25600, dtype=np.float32)])
        mock_load.return_value = np.zeros(25600, dtype=np.float32)
        mock_slice.return_value = ([np.zeros(25600, dtype=np.float32)], [25600])

        for en_post_proc in (False, True):
            with self.subTest(en_post_proc=en_post_proc):
                results_seq = [
                    [{"key": "test_utt", "value": [[0, 1600]]}],
                    [
                        {
                            "text": tag + surface_text,
                            "timestamp": [[i * 100, (i + 1) * 100] for i in range(len(words))],
                            "words": words,
                        }
                    ],
                    [
                        {
                            "text": " Don ' t stop. Https : / / nature .com. Email @ example .com.",
                            "punc_array": [1, 3, 1, 3],
                        }
                    ],
                ]

                def mock_inference(data, *args, **kwargs):
                    if len(results_seq) == 1:
                        self.assertEqual(data, surface_text)
                    return results_seq.pop(0)

                am.inference = MagicMock(side_effect=mock_inference)
                results = am.inference_with_vad(
                    "dummy_input",
                    sentence_timestamp=True,
                    en_post_proc=en_post_proc,
                )

                self.assertEqual(
                    results[0]["text"],
                    "don't stop. https://nature.com email@example.com.",
                )
                self.assertEqual(
                    [(item["start"], item["end"]) for item in results[0]["sentence_info"]],
                    [(0, 400), (400, 1600)],
                )
                self.assertEqual(
                    [item["text"] for item in results[0]["sentence_info"]],
                    ["don't stop.", "https://nature.com email@example.com."],
                )
                self.assertEqual(results[0]["sentence_info"][-1]["timestamp"][-1], [1100, 1600])

    @patch("funasr.auto.auto_model.slice_padding_audio_samples")
    @patch("funasr.auto.auto_model.load_audio_text_image_video")
    @patch("funasr.auto.auto_model.prepare_data_iterator")
    def test_sentence_timestamp_ignores_empty_asr_words(self, mock_prep, mock_load, mock_slice):
        """Malformed word metadata must fall back to the existing aligned text."""
        am = self._make_auto_model(punc_model=MagicMock())
        results_seq = [
            [{"key": "test_utt", "value": [[0, 1000]]}],
            [
                {
                    "text": "你 好",
                    "timestamp": [[0, 500], [500, 1000]],
                    "words": ["你", ""],
                }
            ],
            [{"text": "你好。", "punc_array": [1, 3]}],
        ]
        am.inference = MagicMock(side_effect=lambda *args, **kwargs: results_seq.pop(0))
        mock_prep.return_value = (["test_utt"], [np.zeros(1000, dtype=np.float32)])
        mock_load.return_value = np.zeros(16000, dtype=np.float32)
        mock_slice.return_value = ([np.zeros(16000, dtype=np.float32)], [16000])

        results = am.inference_with_vad("dummy_input", sentence_timestamp=True)

        self.assertEqual(results[0]["sentence_info"][0]["text"], "你好。")

    @patch("funasr.auto.auto_model.slice_padding_audio_samples")
    @patch("funasr.auto.auto_model.load_audio_text_image_video")
    @patch("funasr.auto.auto_model.prepare_data_iterator")
    def test_sentence_timestamp_handles_unsized_punc_array(self, mock_prep, mock_load, mock_slice):
        """Malformed punctuation metadata must use the legacy no-punctuation fallback."""
        am = self._make_auto_model(punc_model=MagicMock())
        results_seq = [
            [{"key": "test_utt", "value": [[0, 1000]]}],
            [
                {
                    "text": "你 好",
                    "timestamp": [[0, 500], [500, 1000]],
                    "words": ["你", "好"],
                }
            ],
            [{"text": "你好。", "punc_array": 3}],
        ]
        am.inference = MagicMock(side_effect=lambda *args, **kwargs: results_seq.pop(0))
        mock_prep.return_value = (["test_utt"], [np.zeros(1000, dtype=np.float32)])
        mock_load.return_value = np.zeros(16000, dtype=np.float32)
        mock_slice.return_value = ([np.zeros(16000, dtype=np.float32)], [16000])

        results = am.inference_with_vad("dummy_input", sentence_timestamp=True)

        self.assertEqual(results[0]["sentence_info"][0]["text"], ["你", "好"])


class TestCTTransformerPunctuation(unittest.TestCase):
    """Keep CT-Transformer text and punctuation-array endings consistent."""

    @patch("funasr.models.ct_transformer.model.load_audio_text_image_video")
    def test_forced_period_uses_sentence_end_id(self, mock_load):
        from funasr.models.ct_transformer.model import CTTransformer

        model = CTTransformer.__new__(CTTransformer)
        torch.nn.Module.__init__(model)
        model.jieba_usr_dict = None
        model.punc_list = ["<unk>", "_", "，", "。", "？", "、"]
        model.sentence_end_id = 3

        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda tokens: np.arange(len(tokens), dtype=np.int64)

        for punc_id in (1, 2, 5):
            for text in ("hello world", "你好"):
                with self.subTest(text=text, punc_id=punc_id):

                    def punc_forward(text, text_lengths, **kwargs):
                        logits = torch.zeros(1, text.shape[1], len(model.punc_list))
                        logits[:, :, punc_id] = 1
                        return logits, None

                    model.punc_forward = punc_forward
                    mock_load.return_value = [text]
                    results, _ = model.inference(
                        data_in=[text],
                        key=["test_utt"],
                        tokenizer=tokenizer,
                        device="cpu",
                        split_size=20,
                    )

                    self.assertTrue(results[0]["text"].endswith((".", "。")))
                    self.assertEqual(int(results[0]["punc_array"][-1]), model.sentence_end_id)


if __name__ == "__main__":
    unittest.main()
