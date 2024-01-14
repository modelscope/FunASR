#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple

from funasr.models.sanm.attention import MultiHeadedAttentionSANM




class MultiHeadedAttentionSANMwithMask(MultiHeadedAttentionSANM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask[0], mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask[1], mask_att_chunk_encoder)
        return att_outs + fsmn_memory


