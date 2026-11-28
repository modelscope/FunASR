import unittest

import numpy as np
import torch

from funasr.utils.load_utils import load_audio_text_image_video


class TestIntegerPcmWaveformInput(unittest.TestCase):
    def test_int32_array_is_scaled_to_float_range(self):
        pcm = np.array([0, 2**30, -(2**30), 2**31 - 1, -(2**31)], dtype=np.int32)
        out = load_audio_text_image_video(pcm)
        self.assertEqual(out.dtype, torch.float32)
        np.testing.assert_allclose(out.numpy(), [0.0, 0.5, -0.5, 1.0, -1.0], atol=1e-6)

    def test_int32_array_matches_equivalent_float_array(self):
        t = np.arange(16000, dtype=np.float32)
        wave = (0.3 * np.sin(t / 15.0)).astype(np.float32)
        pcm = np.round(wave.astype(np.float64) * 2**31).astype(np.int32)
        from_pcm = load_audio_text_image_video(pcm).numpy()
        np.testing.assert_allclose(from_pcm, wave, atol=1e-6)

    def test_int16_tensor_is_scaled_to_float_range(self):
        pcm = torch.tensor([0, 16384, -16384, 32767, -32768], dtype=torch.int16)
        out = load_audio_text_image_video(pcm)
        self.assertTrue(out.is_floating_point())
        np.testing.assert_allclose(
            out.numpy(), [0.0, 0.5, -0.5, 32767 / 32768, -1.0], atol=1e-7
        )

    def test_int32_tensor_is_scaled_to_float_range(self):
        pcm = torch.tensor([0, 2**30, -(2**30), -(2**31)], dtype=torch.int32)
        out = load_audio_text_image_video(pcm)
        self.assertTrue(out.is_floating_point())
        np.testing.assert_allclose(out.numpy(), [0.0, 0.5, -0.5, -1.0], atol=1e-6)

    def test_two_dimensional_int32_array_is_reduced_to_mono(self):
        left = np.full(8000, 2**29, dtype=np.int32)
        right = np.full(8000, 2**30, dtype=np.int32)
        stereo = np.stack([left, right], axis=1)  # (samples, channels)
        out = load_audio_text_image_video(stereo)
        self.assertEqual(tuple(out.shape), (8000,))
        np.testing.assert_allclose(out.numpy(), np.full(8000, 0.375), atol=1e-6)

    def test_integer_data_that_is_not_sound_is_untouched(self):
        ids = np.array([3, 5, 7], dtype=np.int32)
        np.testing.assert_array_equal(
            load_audio_text_image_video(ids, data_type="fbank").numpy(), ids
        )
        tensor_ids = torch.tensor([3, 5, 7], dtype=torch.int16)
        self.assertTrue(
            torch.equal(load_audio_text_image_video(tensor_ids, data_type="fbank"), tensor_ids)
        )

    def test_float_inputs_are_unchanged(self):
        wave = np.array([0.25, -0.5], dtype=np.float32)
        np.testing.assert_array_equal(load_audio_text_image_video(wave).numpy(), wave)
        tensor = torch.tensor([0.25, -0.5])
        self.assertTrue(torch.equal(load_audio_text_image_video(tensor), tensor))


if __name__ == "__main__":
    unittest.main()
