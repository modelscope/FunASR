"""Run against an installed funasr-onnx package."""

import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec("funasr_onnx") is None:
    raise unittest.SkipTest("Install the standalone funasr-onnx distribution first.")

from funasr_onnx.paraformer_online_bin import Paraformer


def make_model(chunk_size):
    model = Paraformer.__new__(Paraformer)
    model.chunk_size = chunk_size
    return model


def make_cache(chunk_size, is_final, last_chunk=False, dims=2):
    left = np.full((1, chunk_size[0] + chunk_size[2], dims), -1.0, dtype=np.float32)
    return {"feats": left, "is_final": is_final, "last_chunk": last_chunk}


def frames(n, dims=2):
    return np.arange(n * dims, dtype=np.float32).reshape(1, n, dims)


class AddOverlapChunkTest(unittest.TestCase):
    def test_zero_left_context_keeps_no_frames_on_final_chunk(self):
        chunk_size = [0, 10, 5]
        model = make_model(chunk_size)
        cache = make_cache(chunk_size, is_final=True)

        model.add_overlap_chunk(frames(10), cache)

        self.assertEqual(cache["feats"].shape[1], 0)

    def test_zero_left_and_right_context_keeps_no_frames(self):
        chunk_size = [0, 10, 0]
        model = make_model(chunk_size)
        cache = make_cache(chunk_size, is_final=False)

        model.add_overlap_chunk(frames(10), cache)

        self.assertEqual(cache["feats"].shape[1], 0)

    def test_nonzero_context_keeps_tail_frames(self):
        chunk_size = [5, 10, 5]
        model = make_model(chunk_size)

        cache = make_cache(chunk_size, is_final=False)
        overlap = model.add_overlap_chunk(frames(10), cache)
        np.testing.assert_array_equal(cache["feats"], overlap[:, -10:, :])

        cache = make_cache(chunk_size, is_final=True)
        overlap = model.add_overlap_chunk(frames(10), cache)
        np.testing.assert_array_equal(cache["feats"], overlap[:, 20 - 5 : 20, :])


if __name__ == "__main__":
    unittest.main()
