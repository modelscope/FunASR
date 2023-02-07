#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        model,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = model.self_attn
        self.feed_forward = model.feed_forward
        self.norm1 = model.norm1
        self.norm2 = model.norm2
        self.size = model.size

    def forward(self, x, mask):

        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        if x.size(2) == residual.size(2):
            x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        if x.size(2) == residual.size(2):
            x = x + residual

        return x, mask



