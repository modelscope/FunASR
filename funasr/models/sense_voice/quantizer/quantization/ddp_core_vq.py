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
import random
import typing as tp
from random import randrange

import numpy as np
from einops import rearrange, repeat
from math import ceil
import torch
from torch import nn
import torch.nn.functional as F
from funasr.utils.hinter import hint_once

from funasr.models.sense_voice.quantizer.quantization import distrib

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float, mask=None):
    if mask is None:
        moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
    else:
        mask = mask.float()
        new_avg = moving_avg * decay + new * (1 - decay)
        new_avg = mask * new_avg + (1 - mask) * moving_avg
        moving_avg.data.copy_(new_avg.data)


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
            sparse_update: bool = False,
            normalized_input: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.decay = decay
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.inited = None
        self.cluster_size = None
        self.embed = None
        self.embed_avg = None
        self.training = True
        self.sparse_update = sparse_update
        self.normalized_input = normalized_input

    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        distrib.broadcast_tensors([self.embed, self.embed_avg, self.cluster_size, self.inited])

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        hint_once(f"threshold_ema_dead_code: {self.threshold_ema_dead_code}.", "threshold_ema_dead_code", rank=0)
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        # sync buffers outside for efficiency
        # distrib.broadcast_tensors(self.buffers())

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

    def encode(self, x, buffers):
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers

        shape = x.shape
        # pre-process
        x = preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind, buffers):
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers

        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x, buffers):
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers

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
            if not self.sparse_update:
                mask = None
            else:
                mask = embed_onehot.sum(0) > 0
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay, mask=mask)
            embed_sum = x.t() @ embed_onehot
            # if self.normalized_input:
            #     embed_sum = F.normalize(embed_sum, dim=0)
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay,
                        mask=mask.unsqueeze(-1) if self.sparse_update else None)
            if not self.sparse_update:
                cluster_size = (
                    laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                    * self.cluster_size.sum()
                )
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            else:
                embed_normalized = self.embed_avg
                if self.normalized_input:
                    embed_normalized = F.normalize(embed_normalized, dim=-1)
            self.embed.data.copy_(embed_normalized)
            # Note: after ema update, there is a very small difference between codebooks on GPUs.
            # The impact can be very small, ignore it.

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
            **kwargs,
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
                                           threshold_ema_dead_code=threshold_ema_dead_code,
                                           **kwargs)
        self.codebook_size = codebook_size
        self.training = True

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x, buffers):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x, buffers)
        return embed_in

    def decode(self, embed_ind, buffers):
        quantize = self._codebook.decode(embed_ind, buffers)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x, buffers):
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x, buffers)

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


class DistributedResidualVectorQuantization(nn.Module):
    """Efficient distributed residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *,
                 num_quantizers,
                 quantize_dropout: bool = False,
                 rand_num_quant: tp.Optional[tp.List] = None,
                 **kwargs):
        super().__init__()
        """
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        """
        codebook_size, codebook_dim = kwargs["codebook_size"], kwargs["dim"]
        kmeans_init = kwargs["kmeans_init"]
        if isinstance(kmeans_init, bool):
            if not kwargs["kmeans_init"]:
                # use uniform init
                embed = uniform_init(num_quantizers, codebook_size, codebook_dim)
                inited = True
                cluster_size = 1
            else:
                # to perform kmeans init on first batch
                embed = torch.zeros(num_quantizers, codebook_size, codebook_dim)
                inited = False
                cluster_size = 0
        elif isinstance(kmeans_init, str):
            # use prepared kmeans init
            embed = np.load(kmeans_init)
            embed = torch.from_numpy(embed)
            if kwargs.get("normalized_input", False):
                logging.info("normalize the code embeddings since the input is normalized.")
                embed = F.normalize(embed, dim=-1)
            if embed.dim() == 2:
                embed = embed.repeat(num_quantizers, 1, 1)
            inited = True
            cluster_size = 1
        else:
            raise TypeError("kmeans_init should be either a bool or string path to init weights.")

        self.register_buffer("inited", torch.Tensor([[inited] for _ in range(num_quantizers)]))
        self.register_buffer("cluster_size", torch.ones(num_quantizers, codebook_size) * cluster_size)
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

        self.q0_ds_ratio = 1
        if "q0_ds_ratio" in kwargs:
            self.q0_ds_ratio = kwargs.pop("q0_ds_ratio")

        self.layers = nn.ModuleList()
        for i in range(num_quantizers):
            vq_args = dict(**kwargs)
            vq = VectorQuantization(**vq_args)
            self.layers.append(vq)

        self.quantize_dropout = quantize_dropout
        self.rand_num_quant = rand_num_quant

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = torch.zeros_like(x)
        residual = x
        bb, cc, tt = x.shape
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
            if should_quantize_dropout and quantizer_index >= rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                all_sub_quants.append(null_sub_quant)
                continue

            quant_in = residual
            if self.q0_ds_ratio > 1 and quantizer_index == 0:
                quant_in = F.interpolate(quant_in, size=[tt//2])
            quantized, indices, loss = layer(quant_in, [
                self.inited[quantizer_index],
                self.cluster_size[quantizer_index],
                self.embed[quantizer_index],
                self.embed_avg[quantizer_index]
            ])
            if self.q0_ds_ratio > 1 and quantizer_index == 0:
                quantized = F.interpolate(quantized, size=[tt])
                indices = F.interpolate(indices.unsqueeze(1).float(), size=[tt]).squeeze(1).long()
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)
            all_sub_quants.append(quantized)

        # sync buffers after one forward step
        distrib.broadcast_tensors(self.buffers())
        out_losses, out_indices, out_sub_quants = map(torch.stack, (all_losses, all_indices, all_sub_quants))

        return quantized_out, out_indices, out_losses, out_sub_quants

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for i, layer in enumerate(self.layers[:n_q]):
            indices = layer.encode(residual, [
                self.inited[i],
                self.cluster_size[i],
                self.embed[i],
                self.embed_avg[i]
            ])
            quantized = layer.decode(indices, [
                self.inited[i],
                self.cluster_size[i],
                self.embed[i],
                self.embed_avg[i]
            ])
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices, [
                self.inited[i],
                self.cluster_size[i],
                self.embed[i],
                self.embed_avg[i]
            ])
            quantized_out = quantized_out + quantized
        return quantized_out
