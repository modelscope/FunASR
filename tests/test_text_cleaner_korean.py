import unittest

from funasr.tokenizer.cleaner import TextCleaner


class TestKoreanTextCleaner(unittest.TestCase):
    def test_korean_cleaner_spells_out_digits_and_latin_letters(self):
        cleaner = TextCleaner("korean_cleaner")
        self.assertEqual(cleaner("  abc 12 "), "에이비씨 일이")

    def test_korean_cleaner_leaves_hangul_unchanged(self):
        cleaner = TextCleaner(["korean_cleaner"])
        self.assertEqual(cleaner("안녕하세요"), "안녕하세요")

    def test_unsupported_cleaner_type_is_still_rejected(self):
        with self.assertRaises(RuntimeError):
            TextCleaner("no_such_cleaner")("text")


if __name__ == "__main__":
    unittest.main()
