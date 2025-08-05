import unittest
import torch
import numpy as np
from funasr.auto.auto_model import AutoModel

class TestAutoModel(unittest.TestCase):

    def setUp(self):
        self.base_kwargs = {
            "model": "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "vad_model": "fsmn-vad",
            "punc_model":"ct-punc", 
            "device": "cpu",
            "batch_size": 1,
            "disable_update": True,
        }

    def test_merge_thr_in_cb_model(self):
        kwargs = self.base_kwargs.copy()
        kwargs["spk_model"] = "cam++"
        merge_thr = 0.5
        kwargs["spk_kwargs"] = {"cb_kwargs": {"merge_thr": merge_thr}}
        model = AutoModel(**kwargs)
        self.assertEqual(model.cb_model.model_config['merge_thr'], merge_thr)
        # res = model.generate(input="/test.wav",
        #              batch_size_s=300)

    def test_progress_callback_called(self):
        class DummyModel:
            def __init__(self):
                self.param = torch.nn.Parameter(torch.zeros(1))

            def parameters(self):
                return iter([self.param])

            def eval(self):
                pass

            def inference(self, data_in=None, **kwargs):
                results = [{"text": str(d)} for d in data_in]
                return results, {"batch_data_time": 1}

        am = AutoModel.__new__(AutoModel)
        am.model = DummyModel()
        am.kwargs = {"batch_size": 2, "disable_pbar": True}

        progress = []

        res = AutoModel.inference(
            am,
            ["a", "b", "c"],
            progress_callback=lambda idx, total: progress.append((idx, total)),
        )

        self.assertEqual(len(progress), 2)
        self.assertEqual(progress, [(2, 3), (3, 3)])


if __name__ == '__main__':
    unittest.main()
