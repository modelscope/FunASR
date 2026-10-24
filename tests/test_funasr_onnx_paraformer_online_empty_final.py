"""Final-result regressions; run with the standalone funasr-onnx dependencies."""

import unittest
from unittest.mock import Mock

import numpy as np

from funasr_onnx.paraformer_online_bin import Paraformer


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


if __name__ == "__main__":
    unittest.main()
