"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""
import random
from math import log2, ceil
from collections import namedtuple
from typing import Optional, List

import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, reduce, pack, unpack

# constants

Return = namedtuple('Return', ['quantized', 'indices', 'entropy_aux_loss'])

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])


# helper functions

def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# entropy

def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)


# class

class LFQ(Module):
    def __init__(
            self,
            *,
            input_size=None,
            dim=None,
            codebook_size=None,
            entropy_loss_weight=0.1,
            commitment_loss_weight=0.25,
            diversity_gamma=1.,
            straight_through_activation="identity",
            num_codebooks=1,
            keep_num_codebooks_dim=None,
            codebook_scale=1.,  # for residual LFQ, codebook scaled down by 2x at each layer
            rand_num_codebooks: Optional[List] = None,
            sampling_rate=16000,
            encoder_hop_length=640,
    ):
        super().__init__()

        # some assert validations
        dim = input_size
        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(
            codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'

        codebook_size = default(codebook_size, lambda: 2 ** dim)
        codebook_dim = int(log2(codebook_size))

        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.project_in = nn.Linear(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # straight through activation

        if straight_through_activation == "identity":
            self.activation = nn.Identity()
        elif straight_through_activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError("Unsupported activation type, only 'tanh' and 'identity' are supported")

        # entropy aux loss related weights

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # for no auxiliary loss, during inference

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent=False)

        # codes

        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.register_buffer('codebook', codebook, persistent=False)
        self.rand_num_codebooks = rand_num_codebooks
        self.sampling_rate = sampling_rate
        self.encoder_hop_length = encoder_hop_length

    def output_size(self):
        return self.dim

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(
            self,
            indices,
            project_out=True
    ):
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... -> ... 1')

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)

        codes = rearrange(codes, '... c d -> ... (c d)')

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def keep_first_nq_codes(self, x, nq=None):
        if nq is None or nq >= self.num_codebooks:
            return x
        inv_p = 1.0 / (nq / self.num_codebooks)
        x[:, :, nq:] = 0
        x[:, :, :nq] = x[:, :, :nq] * inv_p

        return x

    def random_dropout_codes(self, inputs):
        x = torch.clone(inputs)
        rand_num = random.choice(self.rand_num_codebooks)
        return self.keep_first_nq_codes(x, nq=rand_num)

    def cal_num_quant(self, bite_width):
        frame_rate = self.sampling_rate / self.encoder_hop_length
        nq = bite_width / frame_rate / self.codebook_dim
        return nq

    def forward(
            self,
            x,
            inv_temperature=100.,
            return_loss_breakdown=False,
            mask=None,
            bite_width=None,
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = x.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)
        x = self.activation(x)

        # split out number of codebooks

        x = rearrange(x, 'b n (c d) -> b n c d', c=self.num_codebooks)

        # quantize by eq 3.

        original_input = x

        codebook_value = torch.ones_like(x) * self.codebook_scale
        # do quantization
        quantized = torch.where(x > 0, codebook_value, -codebook_value)

        # use straight-through gradients (optionally with custom activation fn) if training

        if self.training:
            x = x + (quantized - x).detach()
        else:
            x = quantized

        # calculate indices

        indices = reduce((x > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

        # entropy aux loss

        if self.training:
            # the same as euclidean distance up to a constant
            distance = -2 * einsum('... i d, j d -> ... i j', original_input, self.codebook)

            prob = (-distance * inv_temperature).softmax(dim=-1)

            per_sample_entropy = entropy(prob).mean()

            # account for mask

            if exists(mask):
                prob = prob[mask]

            # distribution over all available tokens in the batch

            avg_prob = reduce(prob, '... c d -> c d', 'mean')
            codebook_entropy = entropy(avg_prob).mean()

            # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
            # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch

            entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
        else:
            # if not training, just return dummy 0
            entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

        # commit loss

        if self.training:
            commit_loss = F.mse_loss(original_input, quantized.detach(), reduction='none')

            if exists(mask):
                commit_loss = commit_loss[mask]

            commit_loss = commit_loss.mean()
        else:
            commit_loss = self.zero

        # randomly dropout codebooks to fit varying bite width
        if self.training and self.rand_num_codebooks is not None:
            x = self.random_dropout_codes(x)
        if bite_width is not None:
            x = self.keep_first_nq_codes(x, self.cal_num_quant(bite_width))

        # merge back codebook dim

        x = rearrange(x, 'b n c d -> b n (c d)')

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if is_img_or_video:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        # complete aux loss

        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight

        if not return_loss_breakdown:
            return x, indices, aux_loss, None

        return x, indices, aux_loss, dict(
            per_sample_entropy=per_sample_entropy,
            codebook_entropy=codebook_entropy,
            commit_loss=commit_loss
        )

    def inference(
            self,
            x,
            bandwidth: int = None,
    ):
        x, indices, _, _ = self.forward(x, bite_width=bandwidth)

        return x, indices, None


class ScalableLFQ(LFQ):
    def __init__(self, *, input_size=None, dim=None, codebook_size=None, entropy_loss_weight=0.1,
                 commitment_loss_weight=0.25, diversity_gamma=1., straight_through_activation=nn.Identity(),
                 num_codebooks=1, keep_num_codebooks_dim=None, codebook_scale=1.,
                 rand_num_codebooks: Optional[List] = None, sampling_rate=16000, hop_length=640, **kwargs):
        super().__init__(input_size=input_size, dim=dim, codebook_size=codebook_size,
                         entropy_loss_weight=entropy_loss_weight, commitment_loss_weight=commitment_loss_weight,
                         diversity_gamma=diversity_gamma, straight_through_activation=straight_through_activation,
                         num_codebooks=num_codebooks, keep_num_codebooks_dim=keep_num_codebooks_dim,
                         codebook_scale=codebook_scale, rand_num_codebooks=rand_num_codebooks,
                         sampling_rate=sampling_rate, hop_length=hop_length)
        codebook_alpha_conf = kwargs.get("codebook_alpha_conf", None)
        self.init_codebook_alpha(codebook_alpha_conf)

    def init_codebook_alpha(self, codebook_alpha_conf: dict):
        assert codebook_alpha_conf is not None, "codebook_alpha_conf cannot be None"
        name = codebook_alpha_conf.get("name", "constant")
        if name == "constant":
            alphas = codebook_alpha_conf.get("alphas", [1.0] * self.num_codebooks)
            assert len(alphas) == self.num_codebooks, \
                f"the length of codebook alphas {len(alphas)} " \
                f"must match num_codebooks {self.num_codebooks}."
            alphas = np.array(alphas)
        elif name == "exponential":
            temp = codebook_alpha_conf.get("temp", 8.0)
            alphas = np.exp(-np.arange(0, self.num_codebooks) / temp)
        else:
            raise TypeError(f"Unknown codebook alpha type {name}.")
        codebook_alpha = torch.tensor(alphas/alphas.sum(), dtype=torch.float32).reshape(1, 1, -1, 1)
        self.register_buffer("codebook_alpha", codebook_alpha)

    def forward(
            self,
            x,
            inv_temperature=100.,
            return_loss_breakdown=False,
            mask=None,
            bite_width=None
    ):
        """
                einstein notation
                b - batch
                n - sequence (or flattened spatial dimensions)
                d - feature dimension, which is also log2(codebook size)
                c - number of codebook dim
                """

        is_img_or_video = x.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)

        # split out number of codebooks
        x = rearrange(x, 'b n (c d) -> b n c d', c=self.num_codebooks)

        # quantize by eq 3.
        original_input = x
        codebook_value = torch.ones_like(x) * self.codebook_scale
        quantized = torch.where(x > 0, codebook_value, -codebook_value)

        # use straight-through gradients (optionally with custom activation fn) if training
        if self.training:
            x = self.activation(x)
            x = x + (quantized - x).detach()
        else:
            x = quantized

        # calculate indices
        indices = reduce((x > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

        # entropy aux loss
        if self.training:
            # the same as euclidean distance up to a constant
            distance = -2 * einsum('... i d, j d -> ... i j', original_input, self.codebook)
            prob = (-distance * inv_temperature).softmax(dim=-1)
            per_sample_entropy = entropy(prob).mean()

            # account for mask
            if exists(mask):
                prob = prob[mask]

            # distribution over all available tokens in the batch
            avg_prob = reduce(prob, '... c d -> c d', 'mean')
            codebook_entropy = entropy(avg_prob).mean()

            # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
            # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch
            entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
        else:
            # if not training, just return dummy 0
            entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

        # commit loss
        if self.training:
            commit_loss = F.mse_loss(original_input, quantized.detach(), reduction='none')

            if exists(mask):
                commit_loss = commit_loss[mask]

            commit_loss = commit_loss.mean()
        else:
            commit_loss = self.zero

        # randomly dropout codebooks to fit s bite width
        if self.training and self.rand_num_codebooks is not None:
            x = self.random_dropout_codes(x)
        if bite_width is not None:
            x = self.keep_first_nq_codes(x, self.cal_num_quant(bite_width))

        x = x * self.codebook_alpha

        # merge back codebook dim
        x = rearrange(x, 'b n c d -> b n (c d)')

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if is_img_or_video:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        # whether to remove single codebook dim
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        # complete aux loss
        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight

        if not return_loss_breakdown:
            return x, indices, aux_loss, None

        return x, indices, aux_loss, dict(
            per_sample_entropy=per_sample_entropy,
            codebook_entropy=codebook_entropy,
            commit_loss=commit_loss
        )
