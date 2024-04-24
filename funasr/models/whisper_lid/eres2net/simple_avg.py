import torch

from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.modules.nets_utils import make_pad_mask


class SimpleAvg(AbsEncoder):
    def __init__(self, feat_dim):
        super(SimpleAvg, self).__init__()
        self.feat_dim = feat_dim

    def forward(self, x, ilens):
        mask = ~make_pad_mask(ilens, maxlen=x.shape[1]).to(x.device)
        avg_x = (x * mask[:, :, None]).sum(1) / mask.sum(-1)[:, None]
        return avg_x

    def output_size(self) -> int:
        return self.feat_dim
