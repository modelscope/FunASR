import unittest

import numpy as np
import torch

from funasr.utils.load_utils import load_audio_text_image_video


class TestMultichannelWaveformInput(unittest.TestCase):
    def setUp(self):
        t = np.arange(1600, dtype=np.float32)
        self.left = np.sin(t / 20.0).astype(np.float32)
        self.right = 0.5 * np.sin(t / 7.0).astype(np.float32)
        self.mono = (self.left + self.right) / 2

    def _check(self, waveform, **kwargs):
        out = load_audio_text_image_video(waveform, **kwargs)
        self.assertEqual(tuple(out.shape), (1600,))
        np.testing.assert_allclose(out.numpy(), self.mono, atol=1e-6)

    def test_channels_first_ndarray_is_averaged(self):
        self._check(np.stack([self.left, self.right]))

    def test_channels_last_ndarray_is_averaged(self):
        self._check(np.stack([self.left, self.right], axis=1))

    def test_channels_first_tensor_is_averaged(self):
        self._check(torch.from_numpy(np.stack([self.left, self.right])))

    def test_single_channel_2d_input_becomes_1d(self):
        for waveform in (self.left[None, :], self.left[:, None]):
            out = load_audio_text_image_video(waveform)
            np.testing.assert_allclose(out.numpy(), self.left, atol=1e-6)

    def test_stereo_is_reduced_before_resampling(self):
        out = load_audio_text_image_video(
            np.stack([self.left, self.right]), fs=16000, audio_fs=8000
        )
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape[0], 3200)

    def test_reduce_channels_false_keeps_channels(self):
        out = load_audio_text_image_video(
            np.stack([self.left, self.right]), reduce_channels=False
        )
        self.assertEqual(tuple(out.shape), (2, 1600))

    def test_fbank_features_are_not_averaged(self):
        feats = np.random.default_rng(0).standard_normal((50, 80)).astype(np.float32)
        out = load_audio_text_image_video(feats, data_type="fbank")
        self.assertEqual(tuple(out.shape), (50, 80))

    def test_empty_2d_input_becomes_empty_1d(self):
        for shape in ((0, 2), (2, 0)):
            out = load_audio_text_image_video(np.zeros(shape, dtype=np.float32))
            self.assertEqual(tuple(out.shape), (0,))


if __name__ == "__main__":
    unittest.main()
