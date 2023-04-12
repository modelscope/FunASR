import torch


def _sampler(pdf: torch.Tensor, num_samples: int,
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
    mask_idxs = _sampler(pdf, num_masked_spans, device=device)

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
