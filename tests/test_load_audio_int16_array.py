import unittest

import numpy as np

from funasr.utils.load_utils import load_audio_text_image_video


class TestInt16WaveformInput(unittest.TestCase):
    def test_int16_array_is_scaled_to_float_range(self):
        pcm = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        out = load_audio_text_image_video(pcm)
        self.assertTrue(out.is_floating_point())
        np.testing.assert_allclose(
            out.numpy(), [0.0, 0.5, -0.5, 32767 / 32768, -1.0], atol=1e-7
        )

    def test_int16_array_matches_equivalent_float_array(self):
        t = np.arange(16000, dtype=np.float32)
        wave = (0.3 * np.sin(t / 15.0)).astype(np.float32)
        pcm = np.round(wave * 32768.0).astype(np.int16)
        from_pcm = load_audio_text_image_video(pcm).numpy()
        from_float = load_audio_text_image_video(wave).numpy()
        np.testing.assert_allclose(from_pcm, from_float, atol=1 / 32768)

    def test_float_array_is_unchanged(self):
        wave = np.array([0.25, -0.5], dtype=np.float32)
        out = load_audio_text_image_video(wave)
        np.testing.assert_array_equal(out.numpy(), wave)

    def test_int16_array_with_resampling_does_not_fail(self):
        pcm = (np.sin(np.arange(1600) / 10.0) * 10000).astype(np.int16)
        out = load_audio_text_image_video(pcm, fs=16000, audio_fs=8000)
        self.assertEqual(out.shape[0], 3200)
        self.assertLessEqual(float(out.abs().max()), 1.0)

    def test_int16_array_that_is_not_sound_is_untouched(self):
        feats = np.array([3, 5, 7], dtype=np.int16)
        out = load_audio_text_image_video(feats, data_type="fbank")
        np.testing.assert_array_equal(out.numpy(), feats)


if __name__ == "__main__":
    unittest.main()
