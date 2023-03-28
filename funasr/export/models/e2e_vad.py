from enum import Enum
from typing import List, Tuple, Dict, Any

import torch
from torch import nn
import math

from funasr.models.encoder.fsmn_encoder import FSMN
from funasr.export.models.encoder.fsmn_encoder import FSMN as FSMN_export

class E2EVadModel(nn.Module):
    def __init__(self, model,
                max_seq_len=512,
                feats_dim=400,
                model_name='model',
                **kwargs,):
        super(E2EVadModel, self).__init__()
        self.feats_dim = feats_dim
        self.max_seq_len = max_seq_len
        self.model_name = model_name
        if isinstance(model.encoder, FSMN):
            self.encoder = FSMN_export(model.encoder)
        else:
            raise "unsupported encoder"
        

    def forward(self, feats: torch.Tensor, *args, ):

        scores, out_caches = self.encoder(feats, *args)
        return scores, out_caches

    def get_dummy_inputs(self, frame=30):
        speech = torch.randn(1, frame, self.feats_dim)
        in_cache0 = torch.randn(1, 128, 19, 1)
        in_cache1 = torch.randn(1, 128, 19, 1)
        in_cache2 = torch.randn(1, 128, 19, 1)
        in_cache3 = torch.randn(1, 128, 19, 1)
        
        return (speech, in_cache0, in_cache1, in_cache2, in_cache3)

    # def get_dummy_inputs_txt(self, txt_file: str = "/mnt/workspace/data_fbank/0207/12345.wav.fea.txt"):
    #     import numpy as np
    #     fbank = np.loadtxt(txt_file)
    #     fbank_lengths = np.array([fbank.shape[0], ], dtype=np.int32)
    #     speech = torch.from_numpy(fbank[None, :, :].astype(np.float32))
    #     speech_lengths = torch.from_numpy(fbank_lengths.astype(np.int32))
    #     return (speech, speech_lengths)

    def get_input_names(self):
        return ['speech', 'in_cache0', 'in_cache1', 'in_cache2', 'in_cache3']

    def get_output_names(self):
        return ['logits', 'out_cache0', 'out_cache1', 'out_cache2', 'out_cache3']

    def get_dynamic_axes(self):
        return {
            'speech': {
                1: 'feats_length'
            },
        }
