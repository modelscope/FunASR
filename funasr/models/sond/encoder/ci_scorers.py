import torch
from torch.nn import functional as F


class DotScorer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            xs_pad: torch.Tensor,
            spk_emb: torch.Tensor,
    ):
        # xs_pad: B, T, D
        # spk_emb: B, N, D
        scores = torch.matmul(xs_pad, spk_emb.transpose(1, 2))
        return scores

    def convert_tf2torch(self, var_dict_tf, var_dict_torch):
        return {}


class CosScorer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            xs_pad: torch.Tensor,
            spk_emb: torch.Tensor,
    ):
        # xs_pad: B, T, D
        # spk_emb: B, N, D
        scores = F.cosine_similarity(xs_pad.unsqueeze(2), spk_emb.unsqueeze(1), dim=-1)
        return scores

    def convert_tf2torch(self, var_dict_tf, var_dict_torch):
        return {}
