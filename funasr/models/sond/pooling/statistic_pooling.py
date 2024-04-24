import torch
from typing import Tuple
from typing import Union
from funasr.models.transformer.utils.nets_utils import make_non_pad_mask
from torch.nn import functional as F
import math

VAR2STD_EPSILON = 1e-12


class StatisticPooling(torch.nn.Module):
    def __init__(self, pooling_dim: Union[int, Tuple] = 2, eps=1e-12):
        super(StatisticPooling, self).__init__()
        if isinstance(pooling_dim, int):
            pooling_dim = (pooling_dim,)
        self.pooling_dim = pooling_dim
        self.eps = eps

    def forward(self, xs_pad, ilens=None):
        # xs_pad in (Batch, Channel, Time, Frequency)

        if ilens is None:
            masks = torch.ones_like(xs_pad).to(xs_pad)
        else:
            masks = make_non_pad_mask(ilens, xs_pad, length_dim=2).to(xs_pad)
        mean = torch.sum(xs_pad, dim=self.pooling_dim, keepdim=True) / torch.sum(
            masks, dim=self.pooling_dim, keepdim=True
        )
        squared_difference = torch.pow(xs_pad - mean, 2.0)
        variance = torch.sum(squared_difference, dim=self.pooling_dim, keepdim=True) / torch.sum(
            masks, dim=self.pooling_dim, keepdim=True
        )
        for i in reversed(self.pooling_dim):
            mean, variance = torch.squeeze(mean, dim=i), torch.squeeze(variance, dim=i)

        mask = torch.less_equal(variance, self.eps).float()
        variance = (1.0 - mask) * variance + mask * self.eps
        stddev = torch.sqrt(variance)

        stat_pooling = torch.cat([mean, stddev], dim=1)

        return stat_pooling


def statistic_pooling(
    xs_pad: torch.Tensor, ilens: torch.Tensor = None, pooling_dim: Tuple = (2, 3)
) -> torch.Tensor:
    # xs_pad in (Batch, Channel, Time, Frequency)

    if ilens is None:
        seq_mask = torch.ones_like(xs_pad).to(xs_pad)
    else:
        seq_mask = make_non_pad_mask(ilens, xs_pad, length_dim=2).to(xs_pad)
    mean = torch.sum(xs_pad, dim=pooling_dim, keepdim=True) / torch.sum(
        seq_mask, dim=pooling_dim, keepdim=True
    )
    squared_difference = torch.pow(xs_pad - mean, 2.0)
    variance = torch.sum(squared_difference, dim=pooling_dim, keepdim=True) / torch.sum(
        seq_mask, dim=pooling_dim, keepdim=True
    )
    for i in reversed(pooling_dim):
        mean, variance = torch.squeeze(mean, dim=i), torch.squeeze(variance, dim=i)

    value_mask = torch.less_equal(variance, VAR2STD_EPSILON).float()
    variance = (1.0 - value_mask) * variance + value_mask * VAR2STD_EPSILON
    stddev = torch.sqrt(variance)

    stat_pooling = torch.cat([mean, stddev], dim=1)

    return stat_pooling


def windowed_statistic_pooling(
    xs_pad: torch.Tensor,
    ilens: torch.Tensor = None,
    pooling_dim: Tuple = (2, 3),
    pooling_size: int = 20,
    pooling_stride: int = 1,
) -> Tuple[torch.Tensor, int]:
    # xs_pad in (Batch, Channel, Time, Frequency)

    tt = xs_pad.shape[2]
    num_chunk = int(math.ceil(tt / pooling_stride))
    pad = pooling_size // 2
    if len(xs_pad.shape) == 4:
        features = F.pad(xs_pad, (0, 0, pad, pad), "replicate")
    else:
        features = F.pad(xs_pad, (pad, pad), "replicate")
    stat_list = []

    for i in range(num_chunk):
        # B x C
        st, ed = i * pooling_stride, i * pooling_stride + pooling_size
        stat = statistic_pooling(features[:, :, st:ed], pooling_dim=pooling_dim)
        stat_list.append(stat.unsqueeze(2))

    # B x C x T
    return torch.cat(stat_list, dim=2), ilens / pooling_stride
