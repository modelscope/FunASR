"""Lightweight 2-Dimensional Convolution module."""

import numpy
import torch
from torch import nn
import torch.nn.functional as F


MIN_VALUE = float(numpy.finfo(numpy.float32).min)


class LightweightConvolution2D(nn.Module):
    """Lightweight 2-Dimensional Convolution layer.

    This implementation is based on
    https://github.com/pytorch/fairseq/tree/master/fairseq

    Args:
        wshare (int): the number of kernel of convolution
        n_feat (int): the number of features
        dropout_rate (float): dropout_rate
        kernel_size (int): kernel size (length)
        use_kernel_mask (bool): Use causal mask or not for convolution kernel
        use_bias (bool): Use bias term or not.

    """

    def __init__(
        self,
        wshare,
        n_feat,
        dropout_rate,
        kernel_size,
        use_kernel_mask=False,
        use_bias=False,
    ):
        """Construct Lightweight 2-Dimensional Convolution layer."""
        super(LightweightConvolution2D, self).__init__()

        assert n_feat % wshare == 0
        self.wshare = wshare
        self.use_kernel_mask = use_kernel_mask
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.padding_size = int(kernel_size / 2)

        # linear -> GLU -> lightconv -> linear
        self.linear1 = nn.Linear(n_feat, n_feat * 2)
        self.linear2 = nn.Linear(n_feat * 2, n_feat)
        self.act = nn.GLU()

        # lightconv related
        self.weight = nn.Parameter(torch.Tensor(self.wshare, 1, kernel_size).uniform_(0, 1))
        self.weight_f = nn.Parameter(torch.Tensor(1, 1, kernel_size).uniform_(0, 1))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_feat))

        # mask of kernel
        kernel_mask0 = torch.zeros(self.wshare, int(kernel_size / 2))
        kernel_mask1 = torch.ones(self.wshare, int(kernel_size / 2 + 1))
        self.kernel_mask = torch.cat((kernel_mask1, kernel_mask0), dim=-1).unsqueeze(1)

    def forward(self, query, key, value, mask):
        """Forward of 'Lightweight 2-Dimensional Convolution'.

        This function takes query, key and value but uses only query.
        This is just for compatibility with self-attention layer (attention.py)

        Args:
            query (torch.Tensor): (batch, time1, d_model) input tensor
            key (torch.Tensor): (batch, time2, d_model) NOT USED
            value (torch.Tensor): (batch, time2, d_model) NOT USED
            mask (torch.Tensor): (batch, time1, time2) mask

        Return:
            x (torch.Tensor): (batch, time1, d_model) output

        """
        # linear -> GLU -> lightconv -> linear
        x = query
        B, T, C = x.size()
        H = self.wshare

        # first liner layer
        x = self.linear1(x)

        # GLU activation
        x = self.act(x)

        # convolution along frequency axis
        weight_f = F.softmax(self.weight_f, dim=-1)
        weight_f = F.dropout(weight_f, self.dropout_rate, training=self.training)
        weight_new = torch.zeros(B * T, 1, self.kernel_size, device=x.device, dtype=x.dtype).copy_(
            weight_f
        )
        xf = F.conv1d(
            x.view(1, B * T, C), weight_new, padding=self.padding_size, groups=B * T
        ).view(B, T, C)

        # lightconv
        x = x.transpose(1, 2).contiguous().view(-1, H, T)  # B x C x T
        weight = F.dropout(self.weight, self.dropout_rate, training=self.training)
        if self.use_kernel_mask:
            self.kernel_mask = self.kernel_mask.to(x.device)
            weight = weight.masked_fill(self.kernel_mask == 0.0, float("-inf"))
        weight = F.softmax(weight, dim=-1)
        x = F.conv1d(x, weight, padding=self.padding_size, groups=self.wshare).view(B, C, T)
        if self.use_bias:
            x = x + self.bias.view(1, -1, 1)
        x = x.transpose(1, 2)  # B x T x C
        x = torch.cat((x, xf), -1)  # B x T x Cx2

        if mask is not None and not self.use_kernel_mask:
            mask = mask.transpose(-1, -2)
            x = x.masked_fill(mask == 0, 0.0)

        # second linear layer
        x = self.linear2(x)
        return x
