# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""
import logging
import typing as tp
from random import randrange

import numpy as np
from einops import rearrange, repeat
from math import ceil
import torch
from torch import nn
import torch.nn.functional as F
import random

from funasr.models.sense_voice.quantizer.quantization import distrib

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


@torch.no_grad()
def kmeans(samples, num_clusters: int, num_iters: int = 10):
    # device = samples.device
    # samples = samples.cpu()
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        # diffs = rearrange(samples, "n d -> n () d") - rearrange(
        #     means, "c d -> () c d"
        # )
        # dists = -(diffs ** 2).sum(dim=-1)
        dists = -(
                samples.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(samples, means.t())
                + means.t().pow(2).sum(0, keepdim=True)
        )

        buckets = dists.max(dim=-1).indices
        del dists
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    # means = means.to(device)
    return means, bins


def preprocess(x):
    x = rearrange(x, "... d -> (...) d")
    return x


def postprocess_emb(embed_ind, shape):
    return embed_ind.view(*shape[:-1])


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.training = True

    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        distrib.broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        distrib.broadcast_tensors(self.buffers())

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            # Note: after ema update, there is a very small difference between codebooks on GPUs.
            # The impact can be very small, ignore it.

        return quantize, embed_ind


class SimpleEuclideanCodebook(nn.Module):
    """Simple Codebook with Euclidean distance.
    Using gradient to update code embeddings instead of EMA.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: tp.Union[bool, torch.Tensor] = False,
        kmeans_iters: int = 10,
        **kwargs
    ):
        super().__init__()
        if isinstance(kmeans_init, bool):
            if kmeans_init:
                embed = torch.zeros(codebook_size, dim)
                inited = False
            else:
                embed = uniform_init(codebook_size, dim)
                inited = True
        else:
            embed = kmeans_init
            inited = True
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters

        self.embed = nn.Embedding(codebook_size, dim)
        self.embed.weight.data.copy_(embed)
        # self.register_parameter("embed", nn.Parameter(embed, requires_grad=True))
        self.register_buffer("inited", torch.Tensor([inited]))
        self.training = True

    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        distrib.broadcast_tensors(self.buffers())

    def quantize(self, x):
        embed = self.embed.weight.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def dequantize(self, embed_ind):
        quantize = self.embed(embed_ind)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_ind = postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently, supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim)) if requires_projection else (nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim)) if requires_projection else (nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size
        self.training = True

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class SimpleVectorQuantization(nn.Module):
    """Vector quantization implementation with SimpleEuclideanCodebook.
    Currently, supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
        **kwargs
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim)) if requires_projection else (nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim)) if requires_projection else (nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        logging.info(f"commitment_weight: {commitment_weight}, codebook_weight: {codebook_weight}.")

        self._codebook = SimpleEuclideanCodebook(
            dim=_codebook_dim, codebook_size=codebook_size,
            kmeans_init=kmeans_init, kmeans_iters=kmeans_iters
        )
        self.codebook_size = codebook_size
        self.training = True

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            # commit loss for codebook
            if self.codebook_weight > 0:
                codebook_loss = F.mse_loss(quantize, x.detach())
                loss = loss + codebook_loss * self.codebook_weight

            # commit loss for encoder
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *,
                 num_quantizers,
                 quantize_dropout: bool = False,
                 rand_num_quant: tp.Optional[tp.List] = None,
                 **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )
        self.quantize_dropout = quantize_dropout
        self.rand_num_quant = rand_num_quant

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = 0.0
        residual = x
        device = x.device

        all_losses = []
        all_indices = []
        all_sub_quants = []
        n_q = n_q or len(self.layers)

        should_quantize_dropout = self.training and self.quantize_dropout and self.rand_num_quant is not None
        if should_quantize_dropout:
            rand_quantize_dropout_index = random.choice(self.rand_num_quant)

            null_indices_shape = (x.shape[0], x.shape[2])
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)
            null_loss = torch.full((1,), 0., device=device, dtype=x.dtype)
            null_sub_quant = torch.full(x.shape, -1, device=device, dtype=x.dtype)

        for quantizer_index, layer in enumerate(self.layers[:n_q]):
            # dropout except the first quantizer
            if should_quantize_dropout and quantizer_index > 0 and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                all_sub_quants.append(null_sub_quant)
                continue

            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)
            all_sub_quants.append(quantized)

        out_losses, out_indices, out_sub_quants = map(torch.stack, (all_losses, all_indices, all_sub_quants))
        return quantized_out, out_indices, out_losses, out_sub_quants

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


class SimpleResidualVectorQuantization(nn.Module):
    """Simple Residual vector quantization with gradient to
    update codebook  instead of EMA
    """
    def __init__(self, *,
                 num_quantizers,
                 quantize_dropout: bool = False,
                 rand_num_quant: tp.Optional[tp.List] = None,
                 **kwargs):
        super().__init__()
        kmeans_init = raw_kmeans_init = kwargs.pop('kmeans_init', True)
        if isinstance(kmeans_init, str):
            # use prepared kmeans init
            embed = np.load(kmeans_init)
            embed = torch.from_numpy(embed)
            if embed.dim() == 2:
                embed = embed.unsqueeze(0)
            kmeans_init = embed

        self.layers = nn.ModuleList([
            SimpleVectorQuantization(
                kmeans_init=kmeans_init[i] if isinstance(kmeans_init, torch.Tensor) else kmeans_init,
                **kwargs
            ) for i in range(num_quantizers)
        ])
        kwargs["kmeans_init"] = raw_kmeans_init
        self.quantize_dropout = quantize_dropout
        self.rand_num_quant = rand_num_quant

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = 0.0
        residual = x
        device = x.device

        all_losses = []
        all_indices = []
        all_sub_quants = []
        n_q = n_q or len(self.layers)

        should_quantize_dropout = self.training and self.quantize_dropout and self.rand_num_quant is not None
        if should_quantize_dropout:
            rand_quantize_dropout_index = random.choice(self.rand_num_quant)

            null_indices_shape = (x.shape[0], x.shape[2])
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)
            null_loss = torch.full((1,), 0., device=device, dtype=x.dtype)
            null_sub_quant = torch.full(x.shape, -1, device=device, dtype=x.dtype)

        for quantizer_index, layer in enumerate(self.layers[:n_q]):
            # dropout except the first quantizer
            if should_quantize_dropout and quantizer_index > 0 and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                all_sub_quants.append(null_sub_quant)
                continue

            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)
            all_sub_quants.append(quantized)

        out_losses, out_indices, out_sub_quants = map(torch.stack, (all_losses, all_indices, all_sub_quants))
        return quantized_out, out_indices, out_losses, out_sub_quants

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
