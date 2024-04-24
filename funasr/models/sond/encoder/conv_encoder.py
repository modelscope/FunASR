from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.layer_norm import LayerNorm
from funasr.models.encoder.abs_encoder import AbsEncoder
import math
from funasr.models.transformer.utils.repeat import repeat


class EncoderLayer(nn.Module):
    def __init__(
        self,
        input_units,
        num_units,
        kernel_size=3,
        activation="tanh",
        stride=1,
        include_batch_norm=False,
        residual=False,
    ):
        super().__init__()
        left_padding = math.ceil((kernel_size - stride) / 2)
        right_padding = kernel_size - stride - left_padding
        self.conv_padding = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.conv1d = nn.Conv1d(
            input_units,
            num_units,
            kernel_size,
            stride,
        )
        self.activation = self.get_activation(activation)
        if include_batch_norm:
            self.bn = nn.BatchNorm1d(num_units, momentum=0.99, eps=1e-3)
        self.residual = residual
        self.include_batch_norm = include_batch_norm
        self.input_units = input_units
        self.num_units = num_units
        self.stride = stride

    @staticmethod
    def get_activation(activation):
        if activation == "tanh":
            return nn.Tanh()
        else:
            return nn.ReLU()

    def forward(self, xs_pad, ilens=None):
        outputs = self.conv1d(self.conv_padding(xs_pad))
        if self.residual and self.stride == 1 and self.input_units == self.num_units:
            outputs = outputs + xs_pad

        if self.include_batch_norm:
            outputs = self.bn(outputs)

        # add parenthesis for repeat module
        return self.activation(outputs), ilens


class ConvEncoder(AbsEncoder):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Convolution encoder in OpenNMT framework
    """

    def __init__(
        self,
        num_layers,
        input_units,
        num_units,
        kernel_size=3,
        dropout_rate=0.3,
        position_encoder=None,
        activation="tanh",
        auxiliary_states=True,
        out_units=None,
        out_norm=False,
        out_residual=False,
        include_batchnorm=False,
        regularization_weight=0.0,
        stride=1,
        tf2torch_tensor_name_prefix_torch: str = "speaker_encoder",
        tf2torch_tensor_name_prefix_tf: str = "EAND/speaker_encoder",
    ):
        super().__init__()
        self._output_size = num_units

        self.num_layers = num_layers
        self.input_units = input_units
        self.num_units = num_units
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.position_encoder = position_encoder
        self.out_units = out_units
        self.auxiliary_states = auxiliary_states
        self.out_norm = out_norm
        self.activation = activation
        self.out_residual = out_residual
        self.include_batch_norm = include_batchnorm
        self.regularization_weight = regularization_weight
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf
        if isinstance(stride, int):
            self.stride = [stride] * self.num_layers
        else:
            self.stride = stride
        self.downsample_rate = 1
        for s in self.stride:
            self.downsample_rate *= s

        self.dropout = nn.Dropout(dropout_rate)
        self.cnn_a = repeat(
            self.num_layers,
            lambda lnum: EncoderLayer(
                input_units if lnum == 0 else num_units,
                num_units,
                kernel_size,
                activation,
                self.stride[lnum],
                include_batchnorm,
                residual=True if lnum > 0 else False,
            ),
        )

        if self.out_units is not None:
            left_padding = math.ceil((kernel_size - stride) / 2)
            right_padding = kernel_size - stride - left_padding
            self.out_padding = nn.ConstantPad1d((left_padding, right_padding), 0.0)
            self.conv_out = nn.Conv1d(
                num_units,
                out_units,
                kernel_size,
            )

        if self.out_norm:
            self.after_norm = LayerNorm(out_units)

    def output_size(self) -> int:
        return self.num_units

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        inputs = xs_pad
        if self.position_encoder is not None:
            inputs = self.position_encoder(inputs)

        if self.dropout_rate > 0:
            inputs = self.dropout(inputs)

        outputs, _ = self.cnn_a(inputs.transpose(1, 2), ilens)

        if self.out_units is not None:
            outputs = self.conv_out(self.out_padding(outputs))

        outputs = outputs.transpose(1, 2)
        if self.out_norm:
            outputs = self.after_norm(outputs)

        if self.out_residual:
            outputs = outputs + inputs

        return outputs, ilens, None
