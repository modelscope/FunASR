import base64
import importlib.util
import io
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_utils_module():
    module_names = ["funasr", "funasr.download", "funasr.download.file"]
    previous = {name: sys.modules.get(name) for name in module_names}

    funasr_package = types.ModuleType("funasr")
    funasr_package.__path__ = []
    download_package = types.ModuleType("funasr.download")
    download_package.__path__ = []
    download_file = types.ModuleType("funasr.download.file")
    download_file.download_from_url = lambda url: url

    sys.modules["funasr"] = funasr_package
    sys.modules["funasr.download"] = download_package
    sys.modules["funasr.download.file"] = download_file
    try:
        spec = importlib.util.spec_from_file_location(
            "load_utils_audio_bytes_test", ROOT / "funasr/utils/load_utils.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


LOAD_UTILS = _load_utils_module()

# 80 ms, 16 kHz mono, generated without ID3 or Xing metadata by ffmpeg 6.1.1.
NO_ID3_MP3 = base64.b64decode(
    "//NIxAAa0LJcBVlIABsLTlpy05ZMsmWnLloB0Uy2BhEGUUDiEXzRhOvE88TrdOJY1lDSQMoYBCIAFBGIO4CYJgHAGBsEw2TtwUQIECBBBwnB8H8oGJTlA/wcOYgB/WDhzIA/wI7n+jhgHz+BDnfg+BAQBDB//B8H1QJghAAQSMFwa3j/mFYgGAQU1hUCfLMm//NIxBcg+gJkDZ2gADIa5vcxhhUcxwIo5huQh2CQ4GXugbI6BkioGWkgiPkQAyWIBoMXQsSBs0DYYRz/hlkMijHCghC3/kNFyi5SaHOHO//KJFSKmReJox//yKkVMi8XjEul1L//8mi8Yl0upF42EoS//Pf///XYuxb//41DPjII3hgHzBwACWYI5Npk2mcG//NIxBYiUm5MAZ6oABXAdm1MCaSE6GJWZaBm8zAWVAg8DDsXAAA46gGGABlEABoH+F1A6xxid//IwiAuAskT//HGWCcIIdJ///J84VCcOm5P///nDQuHTcvqNC5///+s3N1Ghos3N1Ghz//WD4gCIPiAIg////gwEQuQYFyiD4Eqijt1s/+gX9S2f8CFjqrn//NIxA8hCpaUAZqYAPlgf8AdQXUCtxZpIEW4sIuMgA7jJMuo+OAmyIkTLikklo/KxOF8qk+YIqSUtH8rE4ZlUvoGy6SqK/5cTPF9R8uLPUlUV0lf8vqPmizyaj6Cz1FdJVFdJX/6c+hPJz5wBpK2LsX/hgBmwwCYHDAJgcMKtVaq1VVMQU1FMy4xMDBVVVVV//NIxA0AAANIAcAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
)[:720]


def _sine_pcm(sample_rate, duration=0.1):
    sample_count = round(sample_rate * duration)
    times = np.arange(sample_count, dtype=np.float64) / sample_rate
    return np.round(np.sin(2 * np.pi * 440 * times) * 12000).astype(np.int16)


def _wav_bytes(samples, sample_rate):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return output.getvalue()


def _rifx_bytes(samples, sample_rate):
    big_endian_samples = samples.astype(">i2").tobytes()
    return (
        b"RIFX"
        + struct.pack(">I", 36 + len(big_endian_samples))
        + b"WAVEfmt "
        + struct.pack(">IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack(">I", len(big_endian_samples))
        + big_endian_samples
    )


def _free_format_mp3_bytes():
    data = bytearray(NO_ID3_MP3)
    for offset in (0, 144, 288, 432, 576):
        data[offset + 2] &= 0x0F
    return bytes(data)


class TestLoadAudioBytes(unittest.TestCase):
    def test_decodes_wav_container_without_treating_header_as_pcm(self):
        samples = _sine_pcm(16000)

        actual = LOAD_UTILS.load_bytes(_wav_bytes(samples, 16000))

        expected = samples.astype(np.float32) / 32768.0
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)

    def test_resamples_wav_container_to_16khz(self):
        samples = _sine_pcm(8000)

        actual = LOAD_UTILS.load_bytes(_wav_bytes(samples, 8000))

        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(len(actual), 1600)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(float(np.max(np.abs(actual))), 0.1)

    def test_decodes_big_endian_rifx_wav(self):
        samples = np.array([-32768, -1000, 0, 1000, 32767], dtype=np.int16)

        actual = LOAD_UTILS.load_bytes(_rifx_bytes(samples, 16000))

        expected = samples.astype(np.float32) / 32768.0
        np.testing.assert_array_equal(actual, expected)

    def test_recognizes_large_wave_container_variants(self):
        decoded = np.array([0.25, -0.25], dtype=np.float32)
        for marker in (b"RF64", b"BW64"):
            container = marker + b"\xff\xff\xff\xffWAVEplaceholder"
            with self.subTest(marker=marker):
                with mock.patch.object(
                    LOAD_UTILS, "load_audio_text_image_video", return_value=decoded
                ):
                    actual = LOAD_UTILS.load_bytes(container)
                np.testing.assert_array_equal(actual, decoded)

    def test_preserves_raw_int16_pcm_bytes(self):
        samples = np.array([-32768, -12345, 0, 12345, 32767], dtype=np.int16)

        actual = LOAD_UTILS.load_bytes(samples.tobytes())

        expected = samples.astype(np.float32) / 32768.0
        np.testing.assert_array_equal(actual, expected)

    def test_preserves_raw_pcm_with_mp3_sync_like_first_sample(self):
        raw_pcm = b"\xff\xfb\x00\x00\x39\x30\xc7\xcf"
        samples = np.frombuffer(raw_pcm, dtype=np.int16)

        actual = LOAD_UTILS.load_bytes(raw_pcm)

        expected = samples.astype(np.float32) / 32768.0
        np.testing.assert_array_equal(actual, expected)

    def test_preserves_raw_pcm_with_inconsistent_free_format_sync_headers(self):
        raw_pcm = bytearray(np.arange(160, dtype=np.int16).tobytes())
        for offset in (0, 100, 210):
            raw_pcm[offset : offset + 4] = b"\xff\xfb\x00\x00"
        raw_pcm = bytes(raw_pcm)
        samples = np.frombuffer(raw_pcm, dtype=np.int16)

        actual = LOAD_UTILS.load_bytes(raw_pcm)

        expected = samples.astype(np.float32) / 32768.0
        np.testing.assert_array_equal(actual, expected)

    def test_preserves_raw_pcm_with_non_wave_riff_prefix(self):
        raw_pcm = b"RIFF\x00\x00\x00\x00NOPE\x00\x00\x00\x00"
        samples = np.frombuffer(raw_pcm, dtype=np.int16)

        actual = LOAD_UTILS.load_bytes(raw_pcm)

        expected = samples.astype(np.float32) / 32768.0
        np.testing.assert_array_equal(actual, expected)

    def test_no_id3_mp3_never_falls_back_to_raw_pcm(self):
        with mock.patch.object(
            LOAD_UTILS,
            "load_audio_text_image_video",
            side_effect=RuntimeError("decoder unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "complete supported audio file"):
                LOAD_UTILS.load_bytes(NO_ID3_MP3)

    def test_free_format_mp3_never_falls_back_to_raw_pcm(self):
        with mock.patch.object(
            LOAD_UTILS,
            "load_audio_text_image_video",
            side_effect=RuntimeError("decoder unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "complete supported audio file"):
                LOAD_UTILS.load_bytes(_free_format_mp3_bytes())

    def test_decodes_no_id3_mp3_consistently_with_file_path(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as mp3_file:
            mp3_file.write(NO_ID3_MP3)
            mp3_file.flush()
            expected = LOAD_UTILS.load_audio_text_image_video(mp3_file.name, fs=16000)

        actual = LOAD_UTILS.load_bytes(NO_ID3_MP3)

        if hasattr(expected, "detach"):
            expected = expected.detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)

    def test_rewinds_file_like_audio_between_decoders(self):
        wav_data = _wav_bytes(_sine_pcm(16000), 16000)

        def consume_then_fail(stream):
            stream.read()
            raise RuntimeError("first decoder failed")

        with mock.patch.object(
            LOAD_UTILS.torchaudio, "load", side_effect=consume_then_fail
        ):
            actual = LOAD_UTILS.load_audio_text_image_video(
                io.BytesIO(wav_data), fs=16000
            )

        expected = _sine_pcm(16000).astype(np.float32) / 32768.0
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)

    @unittest.skipUnless(LOAD_UTILS.is_ffmpeg_installed(), "ffmpeg is required")
    def test_ffmpeg_decodes_file_like_audio(self):
        samples = _sine_pcm(16000)

        actual = LOAD_UTILS._load_audio_ffmpeg(
            io.BytesIO(_wav_bytes(samples, 16000)), sr=16000
        )

        expected = samples.astype(np.float32) / 32768.0
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)

    def test_corrupt_container_error_is_actionable(self):
        corrupt_wav = b"RIFF\x10\x00\x00\x00WAVEbroken"

        with self.assertRaisesRegex(RuntimeError, "complete supported audio file"):
            LOAD_UTILS.load_bytes(corrupt_wav)


if __name__ == "__main__":
    unittest.main()
