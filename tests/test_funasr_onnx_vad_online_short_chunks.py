"""Run against an installed funasr-onnx package."""

import importlib.util
import itertools
import unittest

import numpy as np

if importlib.util.find_spec("funasr_onnx") is None:
    raise unittest.SkipTest("Install the standalone funasr-onnx distribution first.")

from funasr_onnx.vad_bin import Fsmn_vad_online

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

# Segments (ms) of make_audio() streamed in 1600-sample chunks, the step used by the demo.
EXPECTED = [[[[220, -1]]], [[[-1, 1660]]], [[[2220, -1]]], [[[-1, 3360]]]]


def fake_infer(inputs):
    # Score loud frames as speech instead of running the ONNX session.
    sil = np.where(inputs[0][0].mean(-1) > 5.0, 0.05, 0.95)
    scores = np.stack((sil, 1.0 - sil), axis=-1)[None].astype(np.float32)
    return scores, inputs[1:]


def make_model():
    # Keep __call__, WavFrontendOnline and E2EVadModel real; no model files.
    model = Fsmn_vad_online.__new__(Fsmn_vad_online)
    model.cmvn_file = None
    # E2EVadModel's defaults equal the template's model_conf.
    model.config = {"frontend_conf": FRONTEND_CONF, "model_conf": {}}
    model.max_end_sil = 800
    model.encoder_conf = {"fsmn_layers": 1, "proj_dim": 2, "lorder": 2}
    model.infer = fake_infer
    return model


def make_audio():
    # 4 s at 16 kHz: noise at 0.5-1.5 s and 2.5-3.2 s, silence elsewhere.
    rng = np.random.default_rng(0)
    audio = np.zeros(64000, dtype=np.float32)
    for start, end in ((8000, 24000), (40000, 51200)):
        audio[start:end] = rng.uniform(-0.3, 0.3, end - start)
    return audio


def stream(chunk_sizes):
    # Feed the audio in chunks, cycling through chunk_sizes; the last chunk is final.
    model = make_model()
    audio = make_audio()
    param_dict = {"in_cache": []}
    results = []
    offset = 0
    for size in itertools.cycle(chunk_sizes):
        chunk = audio[offset : offset + size]
        offset += size
        param_dict["is_final"] = offset >= len(audio)
        segments = model(audio_in=chunk, param_dict=param_dict)
        if segments:
            results.append(segments)
        if param_dict["is_final"]:
            return results


class FsmnVadOnlineShortChunkTest(unittest.TestCase):
    def test_demo_chunk_size(self):
        self.assertEqual(stream([1600]), EXPECTED)

    def test_chunks_shorter_than_first_lfr_frame(self):
        # The first LFR frame needs 720 samples (45 ms), so these first calls yield no features.
        for size in (160, 320, 480):
            with self.subTest(size=size):
                self.assertEqual(stream([size]), EXPECTED)

    def test_packets_that_complete_no_fbank_frame(self):
        # The first 1600-sample chunk leaves 320 samples for the next 400-sample fbank window,
        # so a 40-sample packet, or the second of two 80-sample packets, completes no frame.
        for chunk_sizes in ([1600, 40], [1600, 80, 80]):
            with self.subTest(chunk_sizes=chunk_sizes):
                self.assertEqual(stream(chunk_sizes), EXPECTED)


if __name__ == "__main__":
    unittest.main()
