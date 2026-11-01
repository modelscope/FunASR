"""Regression tests for empty text in the realtime CT-Transformer punctuation model."""

import unittest

import torch

from funasr.auto.auto_model import AutoModel
from funasr.models.ct_transformer_streaming.model import CTTransformerStreaming
from funasr.tokenizer.char_tokenizer import CharTokenizer

PUNC_LIST = ["<unk>", "_", "，", "。", "？", "、"]
TOKEN_LIST = ["<blank>", "<s>", "</s>", "<unk>", "跨", "境", "河", "流", "是"]


def _tiny_streaming_punc_model():
    """Build a small randomly initialised CTTransformerStreaming that never adds punctuation."""
    model = CTTransformerStreaming(
        encoder="SANMVadEncoder",
        encoder_conf={
            "input_size": 8,
            "output_size": 8,
            "attention_heads": 2,
            "linear_units": 16,
            "num_blocks": 2,
            "dropout_rate": 0.0,
            "positional_dropout_rate": 0.0,
            "attention_dropout_rate": 0.0,
            "input_layer": "pe",
            "kernel_size": 3,
            "sanm_shfit": 0,
            "padding_idx": 0,
        },
        vocab_size=len(TOKEN_LIST),
        punc_list=PUNC_LIST,
        embed_unit=8,
        att_unit=8,
        sentence_end_id=3,
    )
    with torch.no_grad():
        model.decoder.weight.zero_()
        model.decoder.bias.zero_()
        model.decoder.bias[PUNC_LIST.index("_")] = 1.0
    return model.eval()


class TestCTTransformerStreamingEmptyInput(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_streaming_punc_model()
        self.tokenizer = CharTokenizer(token_list=TOKEN_LIST, unk_symbol="<unk>")

    def test_empty_text_returns_empty_result(self):
        for text in ("", "   "):
            with self.subTest(text=text):
                results, _ = self.model.inference(
                    [text], key=["utt"], tokenizer=self.tokenizer, cache={}, device="cpu"
                )
                self.assertEqual(results, [{"key": "utt", "text": "", "punc_array": None}])

    def test_empty_segment_keeps_pending_words(self):
        am = AutoModel.__new__(AutoModel)
        am.model = self.model
        am.vad_model = None
        am.punc_model = None
        am.kwargs = {"device": "cpu", "disable_pbar": True, "tokenizer": self.tokenizer}

        cache = {}
        self.assertEqual(am.generate(input="跨境河流", cache=cache)[0]["text"], "跨境河流")
        self.assertEqual(cache["pre_text"], ["跨", "境", "河", "流"])

        self.assertEqual(am.generate(input="", cache=cache)[0]["text"], "")
        self.assertEqual(cache["pre_text"], ["跨", "境", "河", "流"])

        self.assertEqual(am.generate(input="是", cache=cache)[0]["text"], "是")


if __name__ == "__main__":
    unittest.main()
