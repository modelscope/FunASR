import torch
from torch.nn import functional as F


class DotScorer(torch.nn.Module):
    def __init__(self):
        """Initialize DotScorer."""
        super().__init__()

    def forward(
        self,
        xs_pad: torch.Tensor,
        spk_emb: torch.Tensor,
    ):
        # xs_pad: B, T, D
        # spk_emb: B, N, D
        """Forward pass for training.
        
            Args:
                xs_pad: TODO.
                spk_emb: TODO.
            """
        scores = torch.matmul(xs_pad, spk_emb.transpose(1, 2))
        return scores


class CosScorer(torch.nn.Module):
    def __init__(self):
        """Initialize CosScorer."""
        super().__init__()

    def forward(
        self,
        xs_pad: torch.Tensor,
        spk_emb: torch.Tensor,
    ):
        # xs_pad: B, T, D
        # spk_emb: B, N, D
        """Forward pass for training.
        
            Args:
                xs_pad: TODO.
                spk_emb: TODO.
            """
        scores = F.cosine_similarity(xs_pad.unsqueeze(2), spk_emb.unsqueeze(1), dim=-1)
        return scores
