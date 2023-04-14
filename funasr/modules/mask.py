# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Mask module."""

import torch


def sampler(pdf: torch.Tensor, num_samples: int,
             device=torch.device('cpu')) -> torch.Tensor:
    size = pdf.size()
    z = -torch.log(torch.rand(size, device=device))
    _, indices = torch.topk(pdf + z, num_samples)
    return indices


def compute_mask_indices(
        size: torch.Size,
        mask_prob: float,
        mask_length: int,
        min_masks: int = 0,
        device=torch.device('cpu'),
) -> torch.Tensor:

    assert len(size) == 2
    batch_size, seq_length = size

    # compute number of masked span in batch
    num_masked_spans = mask_prob * float(seq_length) / float(
        mask_length) + torch.rand(1)[0]
    num_masked_spans = int(num_masked_spans)
    num_masked_spans = max(num_masked_spans, min_masks)

    # num_masked <= seq_length
    if num_masked_spans * mask_length > seq_length:
        num_masked_spans = seq_length // mask_length

    pdf = torch.ones(batch_size, seq_length - (mask_length - 1), device=device)
    mask_idxs = sampler(pdf, num_masked_spans, device=device)

    mask_idxs = mask_idxs.unsqueeze(-1).repeat(1, 1, mask_length).view(
        batch_size,
        num_masked_spans * mask_length)  # [B,num_masked_spans*mask_length]

    offset = torch.arange(mask_length, device=device).view(1, 1, -1).repeat(
        1, num_masked_spans, 1)  # [1,num_masked_spans,mask_length]
    offset = offset.view(1, num_masked_spans * mask_length)

    mask_idxs = mask_idxs + offset  # [B,num_masked_spans, mask_length]

    ones = torch.ones(batch_size,
                      seq_length,
                      dtype=torch.bool,
                      device=mask_idxs.device)
    # masks to fill
    full_mask = torch.zeros_like(ones,
                                 dtype=torch.bool,
                                 device=mask_idxs.device)
    return torch.scatter(full_mask, dim=1, index=mask_idxs, src=ones)


def subsequent_mask(size, device="cpu", dtype=torch.bool):
    """Create mask for subsequent steps (size, size).

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)


def target_mask(ys_in_pad, ignore_id):
    """Create mask for decoder self-attention.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int ignore_id: index of padding
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor (B, Lmax, Lmax)
    """
    ys_mask = ys_in_pad != ignore_id
    m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
    return ys_mask.unsqueeze(-2) & m

def vad_mask(size, vad_pos, device="cpu", dtype=torch.bool):
    """Create mask for decoder self-attention.

    :param int size: size of mask
    :param int vad_pos: index of vad index
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor (B, Lmax, Lmax)
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    if vad_pos <= 0 or vad_pos >= size:
        return ret
    sub_corner = torch.zeros(
        vad_pos - 1, size - vad_pos, device=device, dtype=dtype)
    ret[0:vad_pos - 1, vad_pos:] = sub_corner
    return ret
