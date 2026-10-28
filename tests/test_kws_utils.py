import re
import unittest

import torch

from funasr.utils.kws_utils import (
    KwsCtcPrefixDecoder,
    query_token_set,
    split_mixed_label,
    symbol_str,
)


class _IdentityCtc:
    def softmax(self, x):
        return x


def _posteriors(token_ids, vocab_size=4):
    # One frame per entry: token_id gets all the probability mass (0 is blank).
    probs = torch.zeros(len(token_ids), vocab_size)
    for frame, token_id in enumerate(token_ids):
        probs[frame, token_id] = 1.0
    return probs


class KwsUtilsTest(unittest.TestCase):
    def test_symbol_regex_strips_punctuation_and_whitespace(self):
        self.assertEqual(re.sub(symbol_str, "", "abc!@# 中文\t★"), "abc中文")

    def test_split_mixed_label_lowercases_ascii_words(self):
        self.assertEqual(split_mixed_label("Hello中文"), ["hello", "中", "文"])

    def test_query_token_set_strips_symbols_before_character_lookup(self):
        symbol_table = {"a": 1, "中": 2, "<unk>": 0}
        tokens, token_ids = query_token_set("A!中", symbol_table, {})

        self.assertEqual(tokens, ("a", "中"))
        self.assertEqual(token_ids, (1, 2))

    def test_empty_keyword_entries_do_not_produce_detections(self):
        token_list = ["<blank>", "<unk>", "小", "云"]
        for keywords in ("小云小云,", "小云小云, ,", ",小云小云", "小云小云,!!"):
            with self.subTest(keywords=keywords):
                decoder = KwsCtcPrefixDecoder(_IdentityCtc(), keywords, token_list, {})

                self.assertEqual(decoder.decode(_posteriors([0, 0, 0, 0])), (False, None, None))
                self.assertEqual(decoder.decode(_posteriors([2, 0, 0, 0])), (False, None, None))
                self.assertEqual(
                    decoder.decode(_posteriors([2, 3, 0, 2, 3])), (True, "小云小云", 1.0)
                )
                self.assertEqual(list(decoder.keywords_token), ["小云小云"])


if __name__ == "__main__":
    unittest.main()
