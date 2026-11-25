"""Exercise the installed standalone package, not a source-directory import."""
import importlib.metadata
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

if importlib.util.find_spec("funasr_torch") is None:
    raise unittest.SkipTest("Install the standalone funasr-torch distribution first.")

import funasr_torch
import numpy as np
import torch
from funasr_torch.paraformer_bin import ContextualParaformer, SeacoParaformer
from funasr_torch.utils.postprocess_utils import abbr_dispose, rich_transcription_postprocess


class InstalledTorchPackageTest(unittest.TestCase):
    def test_hotwords_are_bounded_and_empty_words_are_removed(self):
        for cls in (ContextualParaformer, SeacoParaformer):
            with self.subTest(wrapper=cls.__name__):
                model = cls.__new__(cls)
                model.vocab = {"a": 3, "b": 4}
                matrix, positions = model.proc_hotword(" ab  abababababab ")
                self.assertEqual(matrix.tolist(), [
                    [3, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                    [3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
                self.assertEqual(positions.tolist(), [1, 9, 0])
                for value in ("", "   \t\n"):
                    matrix, positions = model.proc_hotword(value)
                    self.assertEqual(matrix.tolist(), [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                    self.assertEqual(positions.tolist(), [0])

    def test_batched_timestamps_ignore_padding(self):
        model = funasr_torch.Paraformer.__new__(funasr_torch.Paraformer)
        model.batch_size = 2
        model.device_id = -1
        model.device = torch.device("cpu")
        model.plot_timestamp_to = ""
        model.frontend = SimpleNamespace(opts=SimpleNamespace(
            frame_opts=SimpleNamespace(samp_freq=16000)))
        model.load_data = lambda values, fs: values
        model.extract_feat = lambda lengths: (
            torch.zeros((len(lengths), max(lengths), 4)), torch.tensor(lengths))
        model.decode = lambda scores, lens: [["\u4f60", "\u597d"] for _ in range(len(lens))]

        # Only acoustic outputs are synthetic; installed timestamp and batching
        # code must keep the shorter recording independent of padded frames.
        def infer(feats, lengths):
            peaks = torch.zeros((len(lengths), feats.shape[1] * 3))
            for row, length in enumerate(lengths.tolist()):
                positions = np.linspace(5, length * 3 - 6, 3).round().astype(int)
                peaks[row, positions] = 1
            return torch.zeros((len(lengths), 2, 4)), torch.full((len(lengths),), 2), torch.zeros_like(peaks), peaks

        model.ort_infer = infer
        single = [model([length])[0] for length in (17, 50)]
        self.assertEqual([row["timestamp"] for row in single], [
            [[70, 470], [470, 945]], [[70, 670], [1450, 2925]],
        ])
        self.assertEqual(model([17, 50]), single)

    def test_terminal_abbreviation_retains_text_and_timestamps(self):
        self.assertEqual(abbr_dispose(["a", " ", "b"]), ["AB"])
        self.assertEqual(abbr_dispose(["a", " ", "b"], [[0, 100], [100, 200]]),
                         (["AB"], [[0, 200]]))
        self.assertEqual(abbr_dispose(["a", " ", "b", " ", "c"]), ["ABC"])

    def test_rich_transcription_keeps_word_boundaries(self):
        self.assertEqual(rich_transcription_postprocess("<|en|>hello<|en|>world"), "hello world")
        self.assertEqual(rich_transcription_postprocess("<|zh|>\u4f60<|zh|>\u597d"), "\u4f60\u597d")

    def test_installed_version_and_origin(self):
        distribution = importlib.metadata.distribution("funasr-torch")
        self.assertEqual(distribution.version, "0.1.7")
        self.assertEqual(Path(funasr_torch.__file__).resolve().parent,
                         Path(distribution.locate_file("funasr_torch")).resolve())


if __name__ == "__main__":
    unittest.main()
