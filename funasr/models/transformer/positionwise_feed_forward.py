#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch

from funasr.models.transformer.layer_norm import LayerNorm


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class PositionwiseFeedForwardDecoderSANMExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.w_1 = model.w_1
        self.w_2 = model.w_2
        self.activation = model.activation
        self.norm = model.norm

    def forward(self, x):
        x = self.activation(self.w_1(x))
        x = self.w_2(self.norm(x))
        return x
