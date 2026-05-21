from typing import Optional

import torch
import torch.nn as nn

import numpy as np


class MakePadMask(nn.Module):
    def __init__(self, max_seq_len=512, flip=True):
        """Initialize MakePadMask.
        
            Args:
                max_seq_len: TODO.
                flip: TODO.
            """
        super().__init__()
        if flip:
            self.mask_pad = torch.Tensor(1 - np.tri(max_seq_len)).type(torch.bool)
        else:
            self.mask_pad = torch.Tensor(np.tri(max_seq_len)).type(torch.bool)

    def forward(self, lengths, xs=None, length_dim=-1, maxlen=None):
        """Make mask tensor containing indices of padded part.
        This implementation creates the same mask tensor with original make_pad_mask,
        which can be converted into onnx format.
        Dimension length of xs should be 2 or 3.
        """
        if length_dim == 0:
            raise ValueError("length_dim cannot be 0: {}".format(length_dim))

        if xs is not None and len(xs.shape) == 3:
            if length_dim == 1:
                lengths = lengths.unsqueeze(1).expand(*xs.transpose(1, 2).shape[:2])
            else:
                lengths = lengths.unsqueeze(1).expand(*xs.shape[:2])

        if maxlen is not None:
            m = maxlen
        elif xs is not None:
            m = xs.shape[-1]
        else:
            m = torch.max(lengths)

        mask = self.mask_pad[lengths - 1][..., :m].type(torch.float32)

        if length_dim == 1:
            return mask.transpose(1, 2)
        else:
            return mask


class sequence_mask(nn.Module):
    def __init__(self, max_seq_len=512, flip=True):
        """Initialize sequence_mask.
        
            Args:
                max_seq_len: TODO.
                flip: TODO.
            """
        super().__init__()

    def forward(self, lengths, max_seq_len=None, dtype=torch.float32, device=None):
        """Forward pass for training.
        
            Args:
                lengths: TODO.
                max_seq_len: TODO.
                dtype: TODO.
                device: Target device ("cuda:0", "cpu", etc.).
            """
        if max_seq_len is None:
            max_seq_len = lengths.max()
        row_vector = torch.arange(0, max_seq_len, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


def normalize(
    input: torch.Tensor, p: float = 2.0, dim: int = 1, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Normalize.
    
        Args:
            input: Input audio/text data.
            p: TODO.
            dim: TODO.
            out: TODO.
        """
    if out is None:
        denom = input.norm(p, dim, keepdim=True).expand_as(input)
        return input / denom
    else:
        denom = input.norm(p, dim, keepdim=True).expand_as(input)
        return torch.div(input, denom, out=out)


def subsequent_mask(size: torch.Tensor):
    """Subsequent mask.
    
        Args:
            size: TODO.
        """
    return torch.ones(size, size).tril()


def MakePadMask_test():
    """Makepadmask test."""
    feats_length = torch.tensor([10]).type(torch.long)
    mask_fn = MakePadMask()
    mask = mask_fn(feats_length)
    print(mask)


if __name__ == "__main__":
    MakePadMask_test()
