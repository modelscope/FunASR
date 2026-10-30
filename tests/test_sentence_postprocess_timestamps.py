import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
POSTPROCESS_UTILS = (
    "funasr/utils/postprocess_utils.py",
    "runtime/python/onnxruntime/funasr_onnx/utils/postprocess_utils.py",
    "runtime/python/libtorch/funasr_torch/utils/postprocess_utils.py",
)


def _load_sentence_postprocess(relative_path):
    spec = importlib.util.spec_from_file_location(
        relative_path.replace("/", "_").replace(".py", ""), REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sentence_postprocess


class SentencePostprocessTimestampTest(unittest.TestCase):
    def assert_timestamps(self, tokens, expected_words, expected_timestamps):
        timestamps = [[i * 100, (i + 1) * 100] for i in range(len(tokens))]
        for path in POSTPROCESS_UTILS:
            with self.subTest(path=path):
                _, result_timestamps, words = _load_sentence_postprocess(path)(
                    tokens, timestamps
                )
                self.assertEqual(words, expected_words)
                self.assertEqual(result_timestamps, expected_timestamps)

    def test_chinese_words_keep_their_timestamps_after_unk(self):
        self.assert_timestamps(
            ["你", "<unk>", "好", "吗"],
            ["你", "好", "吗"],
            [[0, 100], [200, 300], [300, 400]],
        )

    def test_english_words_keep_their_timestamps_after_unk(self):
        self.assert_timestamps(
            ["hello", "<unk>", "wor@@", "ld"],
            ["hello", "world"],
            [[0, 100], [200, 400]],
        )

    def test_mixed_words_keep_their_timestamps_after_unk(self):
        self.assert_timestamps(
            ["你", "<unk>", "hello", "好"],
            ["你", "hello", "好"],
            [[0, 100], [200, 300], [300, 400]],
        )

    def test_tokens_without_a_timestamp_at_the_end_are_ignored(self):
        for path in POSTPROCESS_UTILS:
            with self.subTest(path=path):
                _, timestamps, words = _load_sentence_postprocess(path)(
                    ["你", "好", "</s>"], [[0, 100], [100, 200]]
                )
                self.assertEqual(words, ["你", "好"])
                self.assertEqual(timestamps, [[0, 100], [100, 200]])

    def test_too_few_timestamps_do_not_crash_when_a_token_is_filtered(self):
        cases = (
            (["你", "<unk>", "好", "吗"], ["你", "好", "吗"]),
            (["hello", "<unk>", "there", "big", "world"], ["hello", "there", "big", "world"]),
            (["你", "<unk>", "hello", "好", "world"], ["你", "hello", "好", "world"]),
        )
        for path in POSTPROCESS_UTILS:
            for tokens, expected_words in cases:
                with self.subTest(path=path, tokens=tokens):
                    timestamps = [[i * 100, (i + 1) * 100] for i in range(len(tokens) - 1)]
                    _, result_timestamps, words = _load_sentence_postprocess(path)(
                        tokens, timestamps
                    )
                    self.assertEqual(words, expected_words)
                    self.assertEqual(len(result_timestamps), len(expected_words))


if __name__ == "__main__":
    unittest.main()
