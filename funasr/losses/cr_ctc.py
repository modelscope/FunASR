"""Consistency-Regularized CTC (CR-CTC) loss.

Based on: "Improving CTC-based Speech Recognition via Consistency Regularization"
Key idea: Run encoder twice (with/without SpecAug), compute KL divergence between
the two CTC outputs as a consistency regularization term.

Usage in training:
    cr_loss = cr_ctc_loss(ctc_logprobs_aug, ctc_logprobs_clean, input_lengths)
    total_loss = ctc_loss + cr_loss_scale * cr_loss
"""

import torch
import torch.nn.functional as F


def cr_ctc_loss(
    log_probs_aug: torch.Tensor,
    log_probs_clean: torch.Tensor,
    input_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute CR-CTC consistency regularization loss.

    Computes symmetric KL divergence between augmented and clean encoder outputs.

    Args:
        log_probs_aug: CTC log probabilities from augmented input (B, T, V)
        log_probs_clean: CTC log probabilities from clean input (B, T, V)
        input_lengths: Valid lengths for each sample (B,)

    Returns:
        Scalar loss value (mean over batch and time).
    """
    batch_size, max_len, _ = log_probs_aug.shape

    # Create mask for valid positions
    mask = torch.arange(max_len, device=input_lengths.device)[None, :] < input_lengths[:, None]
    mask = mask.unsqueeze(-1)  # (B, T, 1)

    # Convert log probs to probs for KL computation
    probs_aug = log_probs_aug.exp()
    probs_clean = log_probs_clean.exp()

    # Symmetric KL divergence: 0.5 * (KL(p||q) + KL(q||p))
    # KL(p||q) = sum(p * (log_p - log_q))
    kl_aug_to_clean = (probs_aug * (log_probs_aug - log_probs_clean)) * mask
    kl_clean_to_aug = (probs_clean * (log_probs_clean - log_probs_aug)) * mask

    # Mean over valid positions
    num_valid = mask.sum()
    if num_valid > 0:
        loss = 0.5 * (kl_aug_to_clean.sum() + kl_clean_to_aug.sum()) / num_valid
    else:
        loss = torch.tensor(0.0, device=log_probs_aug.device)

    return loss
