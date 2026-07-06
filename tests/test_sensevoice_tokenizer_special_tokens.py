"""Unit tests for SenseVoice Tokenizer handling of special-token strings.

Regression guard for issue #3110: ASR models such as Fun-ASR-Nano occasionally
emit special-token strings (e.g. ``<|no|>``, a language tag) as part of the
transcription text. Re-encoding that text via ``tiktoken.Encoding.encode``
crashes by default (``disallowed_special="all"``), which takes down the whole
batch on a single bad sample during forced alignment / loss computation.
``Tokenizer.encode`` now treats special-token strings as ordinary text unless
the caller explicitly opts into a stricter policy.

These tests build a minimal in-memory tiktoken encoding, so they run without
the Fun-ASR-Nano ``multilingual.tiktoken`` vocab (which ships with the model,
not this repo) and without a GPU or model download.
"""

import unittest

import tiktoken

from funasr.models.sense_voice.whisper_lib.tokenizer import Tokenizer


def _build_test_encoding() -> tiktoken.Encoding:
    """A minimal byte-level BPE encoding with the specials Tokenizer needs.

    Every single byte is its own token, so arbitrary text can be encoded
    without the model's multilingual.tiktoken vocab file.
    """
    mergeable_ranks = {bytes([b]): b for b in range(256)}
    n_vocab = len(mergeable_ranks)

    # Specials required by Tokenizer.__post_init__ (startoftranscript/translate/
    # transcribe) plus a few language tags. "<|no|>" is the Norwegian tag that
    # triggered #3110.
    required_specials = [
        "<|startoftranscript|>",
        "<|no|>",
        "<|zh|>",
        "<|en|>",
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        "<|0.00|>",
    ]
    special_tokens = {}
    for tok in required_specials:
        special_tokens[tok] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name="test-encoding",
        pat_str=(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| """
            r"""?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        ),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )


def _build_test_tokenizer() -> Tokenizer:
    return Tokenizer(encoding=_build_test_encoding(), num_languages=1)


class TestTokenizerEncodeSpecialTokens(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _build_test_tokenizer()
        self.no_id = self.tokenizer.special_tokens["<|no|>"]

    def test_plain_text_encodes(self):
        ids = self.tokenizer.encode("hello world")
        self.assertIsInstance(ids, list)
        self.assertTrue(ids)

    def test_language_tag_in_text_does_not_raise(self):
        # The exact failure from issue #3110: ASR output containing "<|no|>"
        # used to crash with "disallowed special token '<|no|>'".
        ids = self.tokenizer.encode("hello <|no|> world")
        self.assertIsInstance(ids, list)
        self.assertTrue(ids)

    def test_nospeech_tag_in_text_does_not_raise(self):
        ids = self.tokenizer.encode("<|nospeech|> something")
        self.assertIsInstance(ids, list)
        self.assertTrue(ids)

    def test_language_tag_alone_does_not_raise(self):
        ids = self.tokenizer.encode("<|no|>")
        self.assertIsInstance(ids, list)
        self.assertTrue(ids)

    def test_caller_disallowed_special_is_respected(self):
        # A caller that explicitly wants the strict behaviour must still get
        # it (setdefault must not override an explicit kwargs).
        with self.assertRaises(ValueError):
            self.tokenizer.encode("<|no|>", disallowed_special="all")

    def test_allowed_special_all_keeps_special_token_ids(self):
        # funasr/models/sense_voice/whisper_lib/decoding.py calls
        # tokenizer.encode(prompt, allowed_special="all"); the special token
        # must still be encoded as its token id, not as raw bytes.
        ids = self.tokenizer.encode("<|no|>", allowed_special="all")
        self.assertEqual(ids, [self.no_id])


if __name__ == "__main__":
    unittest.main()
