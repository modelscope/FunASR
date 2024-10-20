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
if __name__ == '__main__':
    unittest.main()