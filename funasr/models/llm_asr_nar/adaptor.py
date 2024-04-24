import torch
import torch.nn as nn

from funasr.register import tables


@tables.register("adaptor_classes", "Linear")
class Linear(nn.Module):
    def __init__(self, downsample_rate, encoder_dim, llm_dim, ffn_dim: int = 2048, **kwargs):
        super().__init__()
        self.k = downsample_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, self.llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
