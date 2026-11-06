"""Final-result regressions; run with the standalone funasr-onnx dependencies."""

import unittest
from unittest.mock import Mock

import numpy as np

from funasr_onnx.paraformer_online_bin import Paraformer
from funasr_onnx.utils.frontend import WavFrontendOnline, SinusoidalPositionEncoderOnline


class ParaformerOnlineEmptyFinalTest(unittest.TestCase):
    def call_final(self, frames, results):
        # Keep __call__, cache preparation and overlap splitting real; no model files.
        model = Paraformer.__new__(Paraformer)
        model.chunk_size = [5, 10, 5]
        model.encoder_output_size = model.feats_dims = 1
        model.fsmn_layer = 0
        model.pe = Mock()
        model.pe.forward.side_effect = lambda feats, start_idx: feats
        model.extract_feat = Mock(return_value=(
            np.zeros((1, frames, 1), dtype=np.float32),
            np.array([frames], dtype=np.int32),
        ))
        model.infer = Mock(side_effect=results)
        cache = {}
        result = model(
            np.zeros(9600, dtype=np.float32),
            param_dict={"cache": cache, "is_final": True},
        )
        self.assertEqual(model.infer.call_count, len(results))
        self.assertTrue(cache["last_chunk"])
        return result

    def test_split_final_without_results(self):
        result = self.call_final(10, [[], []])
        self.assertEqual(result, [])
        # The official demo consumes any nonempty result this way.
        final_result = ""
        if len(result) > 0:
            final_result += result[0]["preds"][0]
        self.assertEqual(final_result, "")

    def test_split_final_with_only_one_result(self):
        for results in (
            [[{"preds": ("hello", ["hello"])}], []],
            [[], [{"preds": ("hello", ["hello"])}]],
        ):
            with self.subTest(results=results):
                result = self.call_final(10, results)
                self.assertEqual(result, [{"preds": ["hello", ["hello"]]}])
                self.assertIsInstance(result[0]["preds"], list)

    def test_split_final_merges_both_results(self):
        result = self.call_final(10, [
            [{"preds": ("hello", ["hello"])}],
            [{"preds": (" world", ["world"])}],
        ])
        self.assertEqual(result, [{"preds": ["hello world", ["hello", "world"]]}])
        self.assertIsInstance(result[0]["preds"], list)

    def test_single_final_with_result(self):
        expected = [{"preds": ("hello", ["hello"])}]
        result = self.call_final(5, [expected])
        self.assertIs(result, expected)
        self.assertIsInstance(result[0]["preds"], tuple)

    def test_single_final_with_empty_text(self):
        expected = [{"preds": ("", [])}]
        result = self.call_final(5, [expected])
        self.assertIs(result, expected)
        self.assertIsInstance(result[0]["preds"], tuple)


class ParaformerOnlineShortChunkTest(unittest.TestCase):
    @staticmethod
    def make_model():
        # Keep the frontend, positions, cache and overlap real; replace model inference only.
        model = Paraformer.__new__(Paraformer)
        model.chunk_size = [5, 10, 5]
        model.encoder_output_size = 4
        model.feats_dims = 28
        model.fsmn_layer = 0
        model.pe = SinusoidalPositionEncoderOnline()
        model.frontend = WavFrontendOnline(
            n_mels=4, lfr_m=7, lfr_n=6, dither=0.0, cmvn_file=None,
        )
        model.infer = Mock(return_value=[{"preds": ("hello", ["hello"])}])
        return model

    def test_first_chunk_without_lfr_frame_returns_no_result(self):
        # A 25 ms fbank window needs 400 samples; the first LFR frame needs 880.
        for final in (False, True):
            for samples in (0, 1, 160, 320, 399, 400, 480, 560, 879):
                with self.subTest(final=final, samples=samples):
                    model = self.make_model()
                    cache = {}
                    result = model(np.zeros(samples, np.float32),
                                   param_dict={"cache": cache, "is_final": final})
                    self.assertEqual(result, [])
                    self.assertEqual(cache, {})
                    model.infer.assert_not_called()
                    if final:
                        self.assertIsNone(model.frontend.input_cache)
                        self.assertEqual(model.frontend.lfr_splice_cache, [])

    def test_short_chunks_preserve_audio_until_first_lfr_frame(self):
        audio = (0.1 * np.sin(np.arange(960) * 2 * np.pi * 440 / 16000)).astype(np.float32)
        model = self.make_model()
        cache = {}
        for start in range(0, 800, 160):
            self.assertEqual(model(audio[start:start + 160], param_dict={"cache": cache}), [])
        model.infer.assert_not_called()
        self.assertEqual(model(audio[800:], param_dict={"cache": cache}),
                         [{"preds": ("hello", ["hello"])}])
        self.assertEqual(cache["start_idx"], 1)
        reference = self.make_model()
        reference(audio, param_dict={"cache": {}})
        np.testing.assert_allclose(model.infer.call_args.args[0], reference.infer.call_args.args[0])
        np.testing.assert_array_equal(model.infer.call_args.args[1], [11])

    def test_empty_final_flushes_buffered_frontend_frames(self):
        for samples in (400, 480, 560, 879):
            with self.subTest(samples=samples):
                model = self.make_model()
                cache = {}
                self.assertEqual(model(np.zeros(samples, np.float32), param_dict={"cache": cache}), [])
                result = model(np.zeros(0, np.float32), param_dict={"cache": cache, "is_final": True})
                self.assertEqual(result, [{"preds": ("hello", ["hello"])}])
                self.assertTrue(cache["last_chunk"])
                model.infer.assert_called_once()
                self.assertIsNone(model.frontend.input_cache)

    def test_empty_final_still_decodes_existing_overlap_cache(self):
        model = self.make_model()
        cache = {}
        model(np.zeros(9600, np.float32), param_dict={"cache": cache})
        expected_feats = cache["feats"].copy()
        model.infer.reset_mock()
        result = model(np.zeros(0, np.float32), param_dict={"cache": cache, "is_final": True})
        self.assertEqual(result, [{"preds": ("hello", ["hello"])}])
        self.assertTrue(cache["last_chunk"])
        model.infer.assert_called_once()
        np.testing.assert_array_equal(model.infer.call_args.args[0], expected_feats)

    def test_short_final_does_not_contaminate_next_stream(self):
        model = self.make_model()
        self.assertEqual(model(np.ones(320, np.float32), param_dict={"is_final": True}), [])
        model(np.zeros(960, np.float32), param_dict={"cache": {}})
        reference = self.make_model()
        reference(np.zeros(960, np.float32), param_dict={"cache": {}})
        np.testing.assert_array_equal(model.infer.call_args.args[0], reference.infer.call_args.args[0])

    def test_empty_nonfinal_call_preserves_existing_cache(self):
        model = self.make_model()
        cache = {}
        model(np.zeros(9600, np.float32), param_dict={"cache": cache})
        expected_feats = cache["feats"].copy()
        expected_start = cache["start_idx"]
        model.infer.reset_mock()
        self.assertEqual(model(np.zeros(0, np.float32), param_dict={"cache": cache}), [])
        self.assertEqual(cache["start_idx"], expected_start)
        np.testing.assert_array_equal(cache["feats"], expected_feats)
        model.infer.assert_not_called()

    def test_shaped_zero_frame_features_skip_inference(self):
        model = self.make_model()
        model.extract_feat = Mock(return_value=(np.empty((1, 0, 28), np.float32), np.array([0], np.int32)))
        self.assertEqual(model(np.zeros(0, np.float32), param_dict={"is_final": True}), [])
        model.infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
