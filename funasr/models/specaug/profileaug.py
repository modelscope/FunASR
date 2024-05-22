from typing import Tuple, Optional
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn


class ProfileAug(nn.Module):
    """
    Implement the augmentation for profiles including:
    - Split aug: split one profile into two profiles, i.e., main and inaccurate, labels assigned to main
    - Merge aug: merge two profiles into one, labels are also merged into one, the other set to zero
    - Disturb aug: disturb some profile with others to simulate the inaccurate clustering centroids.
    """

    def __init__(
        self,
        apply_split_aug: bool = True,
        split_aug_prob: float = 0.05,
        apply_merge_aug: bool = True,
        merge_aug_prob: float = 0.2,
        apply_disturb_aug: bool = True,
        disturb_aug_prob: float = 0.4,
        disturb_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.apply_split_aug = apply_split_aug
        self.split_aug_prob = split_aug_prob
        self.apply_merge_aug = apply_merge_aug
        self.merge_aug_prob = merge_aug_prob
        self.apply_disturb_aug = apply_disturb_aug
        self.disturb_aug_prob = disturb_aug_prob
        self.disturb_alpha = disturb_alpha

    def split_aug(self, profile: torch.Tensor, binary_labels: torch.Tensor, mask: torch.Tensor):
        # B, N
        bsz, dim = profile.shape[0], profile.shape[-1]
        profile_norm = torch.linalg.norm(profile, dim=-1, keepdim=False)
        spk_count = binary_labels.sum(dim=1)
        prob = np.random.rand(bsz)
        batch_indices = np.nonzero(prob < self.split_aug_prob)[0]
        for idx in batch_indices:
            valid_spk_idx = torch.nonzero(spk_count[idx] * mask[idx])
            pad_spk_idx = torch.nonzero((spk_count[idx] == 0) * mask[idx])
            if len(valid_spk_idx) == 0 or len(pad_spk_idx) == 0:
                continue
            split_spk_idx = valid_spk_idx[torch.randint(len(valid_spk_idx), ())]
            to_cover_idx = pad_spk_idx[torch.randint(len(pad_spk_idx), ())]
            disturb_vec = torch.randn((dim,)).to(profile)
            disturb_vec = F.normalize(disturb_vec, dim=-1)
            profile[idx, to_cover_idx] = F.normalize(
                profile[idx, split_spk_idx] + self.disturb_alpha * disturb_vec
            )
            mask[idx, split_spk_idx] = 0
            mask[idx, to_cover_idx] = 0
        return profile, binary_labels, mask

    def merge_aug(self, profile: torch.Tensor, binary_labels: torch.Tensor, mask: torch.Tensor):
        bsz, dim = profile.shape[0], profile.shape[-1]
        profile_norm = torch.linalg.norm(profile, dim=-1, keepdim=False)
        spk_count = binary_labels.sum(dim=1)
        prob = np.random.rand(bsz)
        batch_indices = np.nonzero(prob < self.merge_aug_prob)[0]
        for idx in batch_indices:
            valid_spk_idx = torch.nonzero(profile_norm[idx] * mask[idx])
            if len(valid_spk_idx) == 0:
                continue
            to_merge = torch.randint(len(valid_spk_idx), (2,))
            spk_idx_1, spk_idx_2 = valid_spk_idx[to_merge[0]], valid_spk_idx[to_merge[1]]
            # merge profile
            profile[idx, spk_idx_1] = profile[idx, spk_idx_1] + profile[idx, spk_idx_2]
            profile[idx, spk_idx_1] = F.normalize(profile[idx, spk_idx_1], dim=-1)
            profile[idx, spk_idx_2] = 0
            # merge binary labels
            binary_labels[idx, :, spk_idx_1] = (
                binary_labels[idx, :, spk_idx_1] + binary_labels[idx, :, spk_idx_2]
            )
            binary_labels[idx, :, spk_idx_1] = (binary_labels[idx, :, spk_idx_1] > 0).to(
                binary_labels
            )
            binary_labels[idx, :, spk_idx_2] = 0

            mask[idx, spk_idx_1] = 0
            mask[idx, spk_idx_2] = 0

        return profile, binary_labels, mask

    def disturb_aug(self, profile: torch.Tensor, binary_labels: torch.Tensor, mask: torch.Tensor):
        bsz, dim = profile.shape[0], profile.shape[-1]
        profile_norm = torch.linalg.norm(profile, dim=-1, keepdim=False)
        spk_count = binary_labels.sum(dim=1)
        prob = np.random.rand(bsz)
        batch_indices = np.nonzero(prob < self.disturb_aug_prob)[0]
        for idx in batch_indices:
            pos_spk_idx = torch.nonzero(spk_count[idx] * mask[idx])
            valid_spk_idx = torch.nonzero(profile_norm[idx] * mask[idx])
            if len(pos_spk_idx) == 0 or len(valid_spk_idx) == 0:
                continue
            to_disturb_idx = pos_spk_idx[torch.randint(len(pos_spk_idx), ())]
            disturb_idx = valid_spk_idx[torch.randint(len(valid_spk_idx), ())]
            alpha = self.disturb_alpha * torch.rand(()).item()
            profile[idx, to_disturb_idx] = (1 - alpha) * profile[
                idx, to_disturb_idx
            ] + alpha * profile[idx, disturb_idx]
            profile[idx, to_disturb_idx] = F.normalize(profile[idx, to_disturb_idx], dim=-1)
            mask[idx, to_disturb_idx] = 0

        return profile, binary_labels, mask

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        profile: torch.Tensor = None,
        profile_lengths: torch.Tensor = None,
        binary_labels: torch.Tensor = None,
        labels_length: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        # copy inputs to avoid inplace-operation
        speech, profile, binary_labels = (
            torch.clone(speech),
            torch.clone(profile),
            torch.clone(binary_labels),
        )
        profile = F.normalize(profile, dim=-1)

        profile_mask = torch.ones(profile.shape[:2]).to(profile)
        if self.apply_disturb_aug:
            profile, binary_labels, profile_mask = self.disturb_aug(
                profile, binary_labels, profile_mask
            )
        if self.apply_split_aug:
            profile, binary_labels, profile_mask = self.split_aug(
                profile, binary_labels, profile_mask
            )
        if self.apply_merge_aug:
            profile, binary_labels, profile_mask = self.merge_aug(
                profile, binary_labels, profile_mask
            )

        return speech, profile, binary_labels
