"""Run against an installed funasr-onnx package."""

import importlib.util
import os
import tempfile
import unittest

import numpy as np

if importlib.util.find_spec("funasr_onnx") is None:
    raise unittest.SkipTest("Install the standalone funasr-onnx distribution first.")

import soundfile
from funasr_onnx.utils.frontend import WavFrontend
from funasr_onnx.vad_bin import Fsmn_vad

# frontend_conf of funasr/models/fsmn_vad_streaming/template.yaml
FRONTEND_CONF = {
    "fs": 16000,
    "window": "hamming",
    "n_mels": 80,
    "frame_length": 25,
    "frame_shift": 10,
    "dither": 0.0,
    "lfr_m": 5,
    "lfr_n": 1,
}

# 16 kHz test files: length and noise ranges in samples, silence elsewhere.
AUDIO = {
    "a": (48000, [(8000, 24000)]),  # 3 s, noise at 0.5-1.5 s
    "b": (72000, [(16000, 32000), (48000, 56000)]),  # 4.5 s, noise at 1-2 s and 3-3.5 s
    "silence": (32000, []),  # 2 s
}

# Segments (ms) of each file on its own.
SEGMENTS = {"a": [[220, 1660]], "b": [[720, 2160], [2720, 3660]], "silence": []}


def fake_infer(inputs):
    # Score loud frames as speech instead of running the ONNX session.
    sil = np.where(inputs[0][0].mean(-1) > 5.0, 0.05, 0.95)
    scores = np.stack((sil, 1.0 - sil), axis=-1)[None].astype(np.float32)
    return scores, inputs[1:]


def make_model(batch_size=1):
    # Keep __call__, WavFrontend and E2EVadModel real; no model files.
    model = Fsmn_vad.__new__(Fsmn_vad)
    model.frontend = WavFrontend(cmvn_file=None, **FRONTEND_CONF)
    model.batch_size = batch_size
    # E2EVadModel's defaults equal the template's model_conf.
    model.vad_scorer_config = {}
    model.max_end_sil = 800
    model.encoder_conf = {"fsmn_layers": 1, "proj_dim": 2, "lorder": 2}
    model.infer = fake_infer
    return model


class FsmnVadMultipleInputsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.wav = {}
        for name, (length, noise) in AUDIO.items():
            rng = np.random.default_rng(0)
            audio = np.zeros(length, dtype=np.float32)
            for start, end in noise:
                audio[start:end] = rng.uniform(-0.3, 0.3, end - start)
            self.wav[name] = os.path.join(tmp.name, name + ".wav")
            soundfile.write(self.wav[name], audio, 16000)

    def test_single_input(self):
        # A path or an array gives a list with one segment list.
        model = make_model()
        for name in AUDIO:
            with self.subTest(name=name):
                self.assertEqual(model(self.wav[name]), [SEGMENTS[name]])
                audio, _ = soundfile.read(self.wav[name], dtype="float32")
                self.assertEqual(model(audio), [SEGMENTS[name]])

    def test_list_input(self):
        # A list of paths gives one segment list per file, in input order.
        model = make_model()
        for names in ([], ["a", "b"], ["b", "a"], ["silence", "b"], ["a", "silence", "b"]):
            with self.subTest(names=names):
                result = model([self.wav[name] for name in names])
                self.assertEqual(result, [SEGMENTS[name] for name in names])

    def test_batch_size_above_one(self):
        # batch_size=2 or 3 gives the batch_size=1 results, for one file and for lists.
        for batch_size in (2, 3):
            model = make_model(batch_size=batch_size)
            for name in AUDIO:
                with self.subTest(batch_size=batch_size, name=name):
                    self.assertEqual(model(self.wav[name]), [SEGMENTS[name]])
            for names in (["a", "b"], ["a", "a"], ["silence", "b"], ["a", "silence", "b"]):
                with self.subTest(batch_size=batch_size, names=names):
                    result = model([self.wav[name] for name in names])
                    self.assertEqual(result, [SEGMENTS[name] for name in names])


if __name__ == "__main__":
    unittest.main()
