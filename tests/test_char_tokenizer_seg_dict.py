"""CharTokenizer must skip blank lines in a seg_dict file instead of raising IndexError."""

import os
import tempfile
import unittest

from funasr.tokenizer.char_tokenizer import CharTokenizer, load_seg_dict


class LoadSegDictBlankLineTest(unittest.TestCase):
    def test_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "seg_dict")
            with open(path, "w", encoding="utf-8") as f:
                f.write("你 你\n\n好 好\n   \nhello he@@ llo\n\n")
            self.assertEqual(load_seg_dict(path), {"你": "你", "好": "好", "hello": "he@@ llo"})
            tokenizer = CharTokenizer(seg_dict=path)
            self.assertEqual(tokenizer.text2tokens("你好 hello"), ["你", "好", "he@@", "llo"])


if __name__ == "__main__":
    unittest.main()
