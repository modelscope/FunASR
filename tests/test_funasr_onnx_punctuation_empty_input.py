"""Run with PYTHONPATH=runtime/python/onnxruntime python -m unittest discover
-s tests -p test_funasr_onnx_punctuation_empty_input.py (standalone ONNX dependencies).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from funasr_onnx.punc_bin import CT_Transformer, CT_Transformer_VadRealtime
from funasr_onnx.utils.utils import TokenIDConverter, code_mix_split_words_jieba


EMPTY_TEXTS = ("", "   ", "\t\n", " \t\r\n\u3000")
PUNC_LIST = ["<unk>", "_", "，", "。", "？", "、"]


def make_model(model_class, segmenter=None):
    # Exercise real tokenization, chunking and cache handling without model downloads.
    model = model_class.__new__(model_class)
    model.seg_jieba = segmenter is not None
    model.code_mix_split_words_jieba = segmenter
    model.converter = TokenIDConverter(["<unk>", "你", "好", "世", "界", "hello", "world"])
    model.punc_list = PUNC_LIST
    model.period = PUNC_LIST.index("。")

    def infer(tokens, lengths, *masks):
        logits = np.zeros((*tokens.shape, len(PUNC_LIST)), dtype=np.float32)
        logits[:, :, PUNC_LIST.index("_")] = 1
        return [logits]

    model.infer = Mock(side_effect=infer)
    return model


class OnnxPunctuationEmptyInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            dictionary = Path(directory) / "jieba_usr_dict"
            dictionary.write_text("你好 100\n世界 100\n", encoding="utf-8")
            cls.segmenter = staticmethod(code_mix_split_words_jieba(str(dictionary)))

    def test_offline_empty_skips_inference(self):
        for segmenter in (None, self.segmenter):
            model = make_model(CT_Transformer, segmenter)
            for text in EMPTY_TEXTS:
                with self.subTest(jieba=segmenter is not None, text=text):
                    self.assertEqual(model(text), ("", []))
                    model.infer.assert_not_called()

    def test_offline_nonempty_around_empty(self):
        for segmenter in (None, self.segmenter):
            for text, expected in (("你好", "你好。"), ("hello world", "hello world。")):
                with self.subTest(jieba=segmenter is not None, text=text):
                    model = make_model(CT_Transformer, segmenter)
                    before = model(text)
                    self.assertEqual(before[0], expected)
                    self.assertEqual(before[1][-1], model.period)
                    for empty in EMPTY_TEXTS:
                        self.assertEqual(model(empty), ("", []))
                    self.assertEqual(model(text), before)
                    self.assertEqual(model.infer.call_count, 2)

    def test_streaming_empty_preserves_cache(self):
        for initial_cache in (None, [], ["你", "好"], ["hello", " world"]):
            with self.subTest(cache=initial_cache):
                model = make_model(CT_Transformer_VadRealtime)
                params = {"cache": initial_cache}
                expected = None if initial_cache is None else initial_cache.copy()
                for text in EMPTY_TEXTS * 2:
                    result = model(text, params)
                    self.assertEqual(result, ("", [], expected or []))
                    self.assertIs(params["cache"], initial_cache)
                    self.assertEqual(params["cache"], expected)
                    if initial_cache is not None:
                        self.assertIs(result[2], initial_cache)
                model.infer.assert_not_called()

    def test_streaming_continuity_across_empty_chunks(self):
        for chunks in (("你好", "世界"), ("hello", "world"), ("你好 hello", "world 世界")):
            for split_size in (2, 20):
                with self.subTest(chunks=chunks, split_size=split_size):
                    model = make_model(CT_Transformer_VadRealtime)
                    reference = make_model(CT_Transformer_VadRealtime)
                    params, reference_params = {"cache": []}, {"cache": []}
                    for chunk in chunks:
                        for empty in EMPTY_TEXTS:
                            cache = params["cache"]
                            calls = model.infer.call_count
                            self.assertEqual(model(empty, params), ("", [], cache))
                            self.assertIs(params["cache"], cache)
                            self.assertEqual(model.infer.call_count, calls)
                        self.assertEqual(
                            model(chunk, params, split_size=split_size),
                            reference(chunk, reference_params, split_size=split_size),
                        )
                        self.assertEqual(params, reference_params)
                    self.assertEqual(model.infer.call_count, reference.infer.call_count)
                    for actual, expected in zip(
                        model.infer.call_args_list, reference.infer.call_args_list
                    ):
                        for actual_arg, expected_arg in zip(actual.args, expected.args):
                            np.testing.assert_array_equal(actual_arg, expected_arg)

    def test_streaming_requires_cache_key_even_for_empty_text(self):
        model = make_model(CT_Transformer_VadRealtime)
        with self.assertRaises(AssertionError):
            model("", {})
        model.infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
