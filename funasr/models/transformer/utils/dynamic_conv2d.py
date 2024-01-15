"""Dynamic 2-Dimensional Convolution module."""

import numpy
import torch
from torch import nn
import torch.nn.functional as F


MIN_VALUE = float(numpy.finfo(numpy.float32).min)


class DynamicConvolution2D(nn.Module):
    """Dynamic 2-Dimensional Convolution layer.

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
        """Construct Dynamic 2-Dimensional Convolution layer."""
        super(DynamicConvolution2D, self).__init__()

        assert n_feat % wshare == 0
        self.wshare = wshare
        self.use_kernel_mask = use_kernel_mask
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.padding_size = int(kernel_size / 2)
        self.attn_t = None
        self.attn_f = None

        # linear -> GLU -- -> lightconv -> linear
        #               \        /
        #                 Linear
        self.linear1 = nn.Linear(n_feat, n_feat * 2)
        self.linear2 = nn.Linear(n_feat * 2, n_feat)
        self.linear_weight = nn.Linear(n_feat, self.wshare * 1 * kernel_size)
        nn.init.xavier_uniform(self.linear_weight.weight)
        self.linear_weight_f = nn.Linear(n_feat, kernel_size)
        nn.init.xavier_uniform(self.linear_weight_f.weight)
        self.act = nn.GLU()

        # dynamic conv related
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_feat))

    def forward(self, query, key, value, mask):
        """Forward of 'Dynamic 2-Dimensional Convolution'.

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
        # linear -> GLU -- -> lightconv -> linear
        #               \        /
        #                 Linear
        x = query
        B, T, C = x.size()
        H = self.wshare
        k = self.kernel_size

        # first liner layer
        x = self.linear1(x)

        # GLU activation
        x = self.act(x)

        # convolution of frequency axis
        weight_f = self.linear_weight_f(x).view(B * T, 1, k)  # B x T x k
        self.attn_f = weight_f.view(B, T, k).unsqueeze(1)
        xf = F.conv1d(
            x.view(1, B * T, C), weight_f, padding=self.padding_size, groups=B * T
        )
        xf = xf.view(B, T, C)

        # get kernel of convolution
        weight = self.linear_weight(x)  # B x T x kH
        weight = F.dropout(weight, self.dropout_rate, training=self.training)
        weight = weight.view(B, T, H, k).transpose(1, 2).contiguous()  # B x H x T x k
        weight_new = torch.zeros(B * H * T * (T + k - 1), dtype=weight.dtype)
        weight_new = weight_new.view(B, H, T, T + k - 1).fill_(float("-inf"))
        weight_new = weight_new.to(x.device)  # B x H x T x T+k-1
        weight_new.as_strided(
            (B, H, T, k), ((T + k - 1) * T * H, (T + k - 1) * T, T + k, 1)
        ).copy_(weight)
        weight_new = weight_new.narrow(-1, int((k - 1) / 2), T)  # B x H x T x T(k)
        if self.use_kernel_mask:
            kernel_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0)
            weight_new = weight_new.masked_fill(kernel_mask == 0.0, float("-inf"))
        weight_new = F.softmax(weight_new, dim=-1)
        self.attn_t = weight_new
        weight_new = weight_new.view(B * H, T, T)

        # convolution
        x = x.transpose(1, 2).contiguous()  # B x C x T
        x = x.view(B * H, int(C / H), T).transpose(1, 2)
        x = torch.bmm(weight_new, x)
        x = x.transpose(1, 2).contiguous().view(B, C, T)

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
