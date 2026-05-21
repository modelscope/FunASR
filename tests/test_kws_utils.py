import re
import unittest

from funasr.utils.kws_utils import query_token_set, split_mixed_label, symbol_str


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


if __name__ == "__main__":
    unittest.main()
