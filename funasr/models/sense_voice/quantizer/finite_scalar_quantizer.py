"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""
import random
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32

from einops import rearrange, pack, unpack


# helper functions

def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# main class

class FSQ(Module):
    def __init__(
            self,
            levels: List[int],
            input_size: Optional[int] = None,
            num_codebooks=1,
            keep_num_codebooks_dim: Optional[bool] = None,
            scale: Optional[float] = None
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(input_size, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def output_size(self):
        return self.dim

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self, z: Tensor, bandwidth: int = None,) -> [Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        commit_loss = torch.tensor([0], dtype=torch.float32, device=z.device)
        return out, indices, commit_loss, None

    def inference(
            self,
            x,
            bandwidth: int = None,
    ):
        x, indices, _, _ = self.forward(x, bandwidth=bandwidth)

        return x, indices, None


class BinaryFSQ(FSQ):
    def __init__(
            self,
            levels: List[int],
            input_size: Optional[int] = None,
            num_codebooks=1,
            keep_num_codebooks_dim: Optional[bool] = None,
            scale: Optional[float] = None,
            rand_num_codebooks: Optional[List] = None,
    ):
        _levels = torch.tensor(levels, dtype=int32)
        assert torch.all(_levels == 2), "BinaryFSQ requires the levels must be 2"
        super().__init__(
            levels, input_size, num_codebooks,
            keep_num_codebooks_dim, scale
        )
        self.rand_num_codebooks = rand_num_codebooks

    def output_size(self):
        return self.dim

    def bound(self, z: Tensor, eps=1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        return torch.sigmoid(z)

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        return quantized

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, '... -> ... 1')
        codes = (indices // self._basis) % self._levels

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self, z: Tensor, bandwidth: int = None,) -> [Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        if self.rand_num_codebooks is not None:
            quant_idx = random.choice(self.rand_num_codebooks)
            codes[:, :, quant_idx:, :] = 0
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        commit_loss = torch.tensor([0], dtype=torch.float32, device=z.device)
        return out, indices, commit_loss, None
