import torch
from torch import nn

class NllLoss(nn.Module):
    """Nll loss.

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function
    """

    def __init__(
        self,
        size,
        padding_idx,
        normalize_length=False,
        criterion=nn.NLLLoss(reduction='none'),
    ):
        """Construct an LabelSmoothingLoss object."""
        super(NllLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
        kl = self.criterion(x , target)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore, 0).sum() / denom
