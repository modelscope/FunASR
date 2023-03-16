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
        self.in_size = model.in_size
        self.size = model.size

    def forward(self, x, mask):

        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        if self.in_size == self.size:
            x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        return x, mask


class EncoderLayerConformer(nn.Module):
    def __init__(
        self,
        model,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = model.self_attn
        self.feed_forward = model.feed_forward
        self.feed_forward_macaron = model.feed_forward_macaron
        self.conv_module = model.conv_module
        self.norm_ff = model.norm_ff
        self.norm_mha = model.norm_mha
        self.norm_ff_macaron = model.norm_ff_macaron
        self.norm_conv = model.norm_conv
        self.norm_final = model.norm_final
        self.size = model.size

    def forward(self, x, mask):
        if isinstance(x, tuple):
            x, pos_emb = x[0], x[1]
        else:
            x, pos_emb = x, None

        if self.feed_forward_macaron is not None:
            residual = x
            x = self.norm_ff_macaron(x)
            x = residual + self.feed_forward_macaron(x) * 0.5

        residual = x
        x = self.norm_mha(x)

        x_q = x

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)
        x = residual + x_att

        if self.conv_module is not None:
            residual = x
            x = self.norm_conv(x)
            x = residual +  self.conv_module(x)

        residual = x
        x = self.norm_ff(x)
        x = residual + self.feed_forward(x) * 0.5

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask
