"""Tokenizers must ignore empty non-linguistic symbols instead of looping forever."""

import os
import tempfile
import unittest

from funasr.tokenizer.char_tokenizer import CharTokenizer
from funasr.tokenizer.phoneme_tokenizer import PhonemeTokenizer


class CharTokenizerNonLinguisticSymbolsTest(unittest.TestCase):
    def test_blank_line_in_symbol_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nlsyms.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("<noise>\n\n")
            for remove in (False, True):
                with self.subTest(remove_non_linguistic_symbols=remove):
                    tokenizer = CharTokenizer(
                        non_linguistic_symbols=path, remove_non_linguistic_symbols=remove
                    )
                    self.assertEqual(tokenizer.non_linguistic_symbols, {"<noise>"})
                    expected = ["你", "好"] if remove else ["你", "<noise>", "好"]
                    self.assertEqual(tokenizer.text2tokens("你<noise>好"), expected)

    def test_empty_symbol_in_iterable_is_ignored(self):
        tokenizer = CharTokenizer(non_linguistic_symbols=["", "<noise>"])
        self.assertEqual(tokenizer.non_linguistic_symbols, {"<noise>"})
        self.assertEqual(tokenizer.text2tokens("a <noise>b"), ["a", "<noise>", "b"])


class PhonemeTokenizerNonLinguisticSymbolsTest(unittest.TestCase):
    def test_empty_symbol_is_ignored(self):
        tokenizer = PhonemeTokenizer(
            g2p_type=None,
            non_linguistic_symbols=["", "<noise>"],
            remove_non_linguistic_symbols=True,
        )
        self.assertEqual(tokenizer.non_linguistic_symbols, {"<noise>"})
        self.assertEqual(tokenizer.text2tokens("a<noise>b"), ["ab"])


if __name__ == "__main__":
    unittest.main()
