import unittest

import torch

from funasr.auto.auto_model import AutoModel


class _ZeroSpeechTimeModel:
    """Reports no speech time, as models do for input shorter than one feature frame."""

    def __init__(self):
        self.param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self.param])

    def inference(self, data_in, **kwargs):
        return [{"key": d, "value": []} for d in data_in], {"batch_data_time": 0.0}


def _auto_model(disable_pbar):
    am = AutoModel.__new__(AutoModel)
    am.model = _ZeroSpeechTimeModel()
    am.kwargs = {"batch_size": 1, "disable_pbar": disable_pbar}
    return am


class TestInferenceZeroSpeechTime(unittest.TestCase):
    def test_zero_speech_time_returns_results_without_progress_bar(self):
        res = AutoModel.inference(_auto_model(True), ["a", "b"])
        self.assertEqual(res, [{"key": "a", "value": []}, {"key": "b", "value": []}])

    def test_zero_speech_time_returns_results_with_progress_bar(self):
        res = AutoModel.inference(_auto_model(False), ["a"])
        self.assertEqual(res, [{"key": "a", "value": []}])


if __name__ == "__main__":
    unittest.main()
