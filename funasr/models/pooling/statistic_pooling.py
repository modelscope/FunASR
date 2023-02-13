import torch
from typing import Tuple
from typing import Union
from funasr.modules.nets_utils import make_non_pad_mask


class StatisticPooling(torch.nn.Module):
    def __init__(self, pooling_dim: Union[int, Tuple] = 2, eps=1e-12):
        super(StatisticPooling, self).__init__()
        if isinstance(pooling_dim, int):
            pooling_dim = (pooling_dim, )
        self.pooling_dim = pooling_dim
        self.eps = eps

    def forward(self, xs_pad, ilens=None):
        # xs_pad in (Batch, Channel, Time, Frequency)

        if ilens is None:
            masks = torch.ones_like(xs_pad).to(xs_pad)
        else:
            masks = make_non_pad_mask(ilens, xs_pad, length_dim=2).to(xs_pad)
        mean = (torch.sum(xs_pad, dim=self.pooling_dim, keepdim=True) /
                torch.sum(masks, dim=self.pooling_dim, keepdim=True))
        squared_difference = torch.pow(xs_pad - mean, 2.0)
        variance = (torch.sum(squared_difference, dim=self.pooling_dim, keepdim=True) /
                    torch.sum(masks, dim=self.pooling_dim, keepdim=True))
        for i in reversed(self.pooling_dim):
            mean, variance = torch.squeeze(mean, dim=i), torch.squeeze(variance, dim=i)

        mask = torch.less_equal(variance, self.eps).float()
        variance = (1.0 - mask) * variance + mask * self.eps
        stddev = torch.sqrt(variance)

        stat_pooling = torch.cat([mean, stddev], dim=1)

        return stat_pooling
