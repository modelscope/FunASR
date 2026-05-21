import torch

from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.modules.nets_utils import make_pad_mask


class SimpleAvg(AbsEncoder):
    def __init__(self, feat_dim):
        """Initialize SimpleAvg.
        
            Args:
                feat_dim: Size/dimension parameter.
            """
        super(SimpleAvg, self).__init__()
        self.feat_dim = feat_dim

    def forward(self, x, ilens):
        """Forward pass for training.
        
            Args:
                x: TODO.
                ilens: TODO.
            """
        mask = ~make_pad_mask(ilens, maxlen=x.shape[1]).to(x.device)
        avg_x = (x * mask[:, :, None]).sum(1) / mask.sum(-1)[:, None]
        return avg_x

    def output_size(self) -> int:
        """Output size."""
        return self.feat_dim
