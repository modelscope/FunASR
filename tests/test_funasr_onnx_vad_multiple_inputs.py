"""Run against an installed funasr-onnx package."""

import importlib.util
import os
import tempfile
import unittest
from unittest.mock import Mock

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


def make_model(batch_size=1, **frontend_conf):
    # Keep __call__, WavFrontend and E2EVadModel real; no model files.
    model = Fsmn_vad.__new__(Fsmn_vad)
    model.frontend = WavFrontend(cmvn_file=None, **{**FRONTEND_CONF, **frontend_conf})
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

    def test_short_inputs_and_frame_boundary(self):
        for fs, frame_length, lfr_m in (
            (16000, 25, 5),
            (8000, 25, 5),
            (16000, 20, 5),
            (16000, 25, 1),
        ):
            window = int(fs * frame_length / 1000)
            for batch_size in (1, 2, 3):
                model = make_model(batch_size, fs=fs, frame_length=frame_length, lfr_m=lfr_m)
                model.infer = Mock(side_effect=fake_infer)
                for samples in (0, 1, window // 2, window - 1, window, window + 1, 2 * window):
                    audio = np.zeros(samples, dtype=np.float32)
                    path = os.path.join(os.path.dirname(self.wav["a"]), "short.wav")
                    soundfile.write(path, audio, fs)
                    for audio_in in (audio, path):
                        with self.subTest(
                            fs=fs,
                            frame_length=frame_length,
                            lfr_m=lfr_m,
                            batch_size=batch_size,
                            samples=samples,
                            input_type=type(audio_in).__name__,
                        ):
                            model.infer.reset_mock()
                            self.assertEqual(model(audio_in), [[]])
                            self.assertEqual(model.infer.call_count, int(samples >= window))

    def test_short_files_keep_their_input_positions(self):
        for samples in (0, 160, 320, 399):
            path = os.path.join(os.path.dirname(self.wav["a"]), "short.wav")
            soundfile.write(path, np.zeros(samples, dtype=np.float32), 16000)
            paths = {**self.wav, "short": path}
            expected = {**SEGMENTS, "short": []}
            for batch_size in (1, 2, 3):
                model = make_model(batch_size)
                model.infer = Mock(side_effect=fake_infer)
                for names in (
                    ["short", "a"],
                    ["a", "short"],
                    ["a", "short", "b"],
                    ["short", "short"],
                    ["short", "b", "short"],
                ):
                    with self.subTest(samples=samples, batch_size=batch_size, names=names):
                        model.infer.reset_mock()
                        result = model([paths[name] for name in names])
                        self.assertEqual(result, [expected[name] for name in names])
                        self.assertEqual(
                            model.infer.call_count, sum(name != "short" for name in names)
                        )
                        self.assertEqual(len({id(segments) for segments in result}), len(result))

    def test_extract_feat_with_empty_frames(self):
        for lfr_m, lfr_n in ((1, 1), (5, 1), (3, 2)):
            model = make_model(lfr_m=lfr_m, lfr_n=lfr_n)
            empty = np.zeros(0, dtype=np.float32)
            short = np.zeros(399, dtype=np.float32)
            normal = np.zeros(1600, dtype=np.float32)
            expected, expected_len = model.extract_feat([normal])
            for waveforms, lengths in (
                ([empty], [0]),
                ([short], [0]),
                ([empty, short, normal], [0, 0, expected_len[0]]),
            ):
                with self.subTest(lfr_m=lfr_m, lfr_n=lfr_n, lengths=lengths):
                    feats, feats_len = model.extract_feat(waveforms)
                    np.testing.assert_array_equal(feats_len, lengths)
                    self.assertEqual(feats.shape, (len(waveforms), max(lengths), 80 * lfr_m))
                    self.assertEqual(feats.dtype, np.float32)
                    if len(waveforms) > 1:
                        np.testing.assert_array_equal(feats[-1], expected[0])
                        self.assertFalse(feats[:-1].any())


if __name__ == "__main__":
    unittest.main()
