#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn
from typing import Optional, Tuple

import torch.nn.functional as F
from funasr.models.transformer.utils.nets_utils import make_pad_mask
import funasr.models.lora.layers as lora


class CosineDistanceAttention(nn.Module):
    """Compute Cosine Distance between spk decoder output and speaker profile
    Args:
        profile_path: speaker profile file path (.npy file)
    """

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, spk_decoder_out, profile, profile_lens=None):
        """
        Args:
            spk_decoder_out(torch.Tensor):(B, L, D)
            spk_profiles(torch.Tensor):(B, N, D)
        """
        x = spk_decoder_out.unsqueeze(2)  # (B, L, 1, D)
        if profile_lens is not None:

            mask = (make_pad_mask(profile_lens)[:, None, :]).to(profile.device)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=x.dtype).numpy().dtype).min)
            weights_not_softmax = F.cosine_similarity(x, profile.unsqueeze(1), dim=-1).masked_fill(
                mask, min_value
            )
            weights = self.softmax(weights_not_softmax).masked_fill(mask, 0.0)  # (B, L, N)
        else:
            x = x[:, -1:, :, :]
            weights_not_softmax = F.cosine_similarity(x, profile.unsqueeze(1).to(x.device), dim=-1)
            weights = self.softmax(weights_not_softmax)  # (B, 1, N)
        spk_embedding = torch.matmul(weights, profile.to(weights.device))  # (B, L, D)

        return spk_embedding, weights
