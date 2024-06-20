#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn
from typing import Optional, Tuple

import torch.nn.functional as F
from funasr.models.transformer.utils.nets_utils import make_pad_mask
import funasr.models.lora.layers as lora


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            min_value = -float(
                "inf"
            )  # min_value = float(np.finfo(torch.tensor(0, dtype=qk.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class MultiHeadedAttentionExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.d_k = model.d_k
        self.h = model.h
        self.linear_q = model.linear_q
        self.linear_k = model.linear_k
        self.linear_v = model.linear_v
        self.linear_out = model.linear_out
        self.attn = None
        self.all_head_size = self.h * self.d_k

    def forward(self, query, key, value, mask):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward_qkv(self, query, key, value):
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        scores = scores + mask

        self.attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(self.attn, value)  # (batch, head, time1, d_k)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.linear_out(context_layer)  # (batch, time1, d_model)


class RelPosMultiHeadedAttentionExport(MultiHeadedAttentionExport):
    def __init__(self, model):
        super().__init__(model)
        self.linear_pos = model.linear_pos
        self.pos_bias_u = model.pos_bias_u
        self.pos_bias_v = model.pos_bias_v

    def forward(self, query, key, value, pos_emb, mask):
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        p = self.transpose_for_scores(self.linear_pos(pos_emb))  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)

    def rel_shift(self, x):
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2
        return x

    def forward_attention(self, value, scores, mask):
        scores = scores + mask

        self.attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(self.attn, value)  # (batch, head, time1, d_k)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.linear_out(context_layer)  # (batch, time1, d_model)


class LegacyRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttentionChunk(torch.nn.Module):
    """RelPositionMultiHeadedAttention definition.
    Args:
        num_heads: Number of attention heads.
        embed_size: Embedding size.
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        num_heads: int,
        embed_size: int,
        dropout_rate: float = 0.0,
        simplified_attention_score: bool = False,
    ) -> None:
        """Construct an MultiHeadedAttention object."""
        super().__init__()

        self.d_k = embed_size // num_heads
        self.num_heads = num_heads

        assert self.d_k * num_heads == embed_size, (
            "embed_size (%d) must be divisible by num_heads (%d)",
            (embed_size, num_heads),
        )

        self.linear_q = torch.nn.Linear(embed_size, embed_size)
        self.linear_k = torch.nn.Linear(embed_size, embed_size)
        self.linear_v = torch.nn.Linear(embed_size, embed_size)

        self.linear_out = torch.nn.Linear(embed_size, embed_size)

        if simplified_attention_score:
            self.linear_pos = torch.nn.Linear(embed_size, num_heads)

            self.compute_att_score = self.compute_simplified_attention_score
        else:
            self.linear_pos = torch.nn.Linear(embed_size, embed_size, bias=False)

            self.pos_bias_u = torch.nn.Parameter(torch.Tensor(num_heads, self.d_k))
            self.pos_bias_v = torch.nn.Parameter(torch.Tensor(num_heads, self.d_k))
            torch.nn.init.xavier_uniform_(self.pos_bias_u)
            torch.nn.init.xavier_uniform_(self.pos_bias_v)

            self.compute_att_score = self.compute_attention_score

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.attn = None

    def rel_shift(self, x: torch.Tensor, left_context: int = 0) -> torch.Tensor:
        """Compute relative positional encoding.
        Args:
            x: Input sequence. (B, H, T_1, 2 * T_1 - 1)
            left_context: Number of frames in left context.
        Returns:
            x: Output sequence. (B, H, T_1, T_2)
        """
        batch_size, n_heads, time1, n = x.shape
        time2 = time1 + left_context

        batch_stride, n_heads_stride, time1_stride, n_stride = x.stride()

        return x.as_strided(
            (batch_size, n_heads, time1, time2),
            (batch_stride, n_heads_stride, time1_stride - n_stride, n_stride),
            storage_offset=(n_stride * (time1 - 1)),
        )

    def compute_simplified_attention_score(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        pos_enc: torch.Tensor,
        left_context: int = 0,
    ) -> torch.Tensor:
        """Simplified attention score computation.
        Reference: https://github.com/k2-fsa/icefall/pull/458
        Args:
            query: Transformed query tensor. (B, H, T_1, d_k)
            key: Transformed key tensor. (B, H, T_2, d_k)
            pos_enc: Positional embedding tensor. (B, 2 * T_1 - 1, size)
            left_context: Number of frames in left context.
        Returns:
            : Attention score. (B, H, T_1, T_2)
        """
        pos_enc = self.linear_pos(pos_enc)

        matrix_ac = torch.matmul(query, key.transpose(2, 3))

        matrix_bd = self.rel_shift(
            pos_enc.transpose(1, 2).unsqueeze(2).repeat(1, 1, query.size(2), 1),
            left_context=left_context,
        )

        return (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

    def compute_attention_score(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        pos_enc: torch.Tensor,
        left_context: int = 0,
    ) -> torch.Tensor:
        """Attention score computation.
        Args:
            query: Transformed query tensor. (B, H, T_1, d_k)
            key: Transformed key tensor. (B, H, T_2, d_k)
            pos_enc: Positional embedding tensor. (B, 2 * T_1 - 1, size)
            left_context: Number of frames in left context.
        Returns:
            : Attention score. (B, H, T_1, T_2)
        """
        p = self.linear_pos(pos_enc).view(pos_enc.size(0), -1, self.num_heads, self.d_k)

        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.permute(0, 2, 3, 1))
        matrix_bd = self.rel_shift(matrix_bd, left_context=left_context)

        return (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.
        Args:
            query: Query tensor. (B, T_1, size)
            key: Key tensor. (B, T_2, size)
            v: Value tensor. (B, T_2, size)
        Returns:
            q: Transformed query tensor. (B, H, T_1, d_k)
            k: Transformed key tensor. (B, H, T_2, d_k)
            v: Transformed value tensor. (B, H, T_2, d_k)
        """
        n_batch = query.size(0)

        q = self.linear_q(query).view(n_batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(n_batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(n_batch, -1, self.num_heads, self.d_k).transpose(1, 2)

        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention context vector.
        Args:
            value: Transformed value. (B, H, T_2, d_k)
            scores: Attention score. (B, H, T_1, T_2)
            mask: Source mask. (B, T_2)
            chunk_mask: Chunk mask. (T_1, T_1)
        Returns:
           attn_output: Transformed value weighted by attention score. (B, T_1, H * d_k)
        """
        batch_size = scores.size(0)
        mask = mask.unsqueeze(1).unsqueeze(2)
        if chunk_mask is not None:
            mask = chunk_mask.unsqueeze(0).unsqueeze(1) | mask
        scores = scores.masked_fill(mask, float("-inf"))
        self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)

        attn_output = self.dropout(self.attn)
        attn_output = torch.matmul(attn_output, value)

        attn_output = self.linear_out(
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        )

        return attn_output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
        left_context: int = 0,
    ) -> torch.Tensor:
        """Compute scaled dot product attention with rel. positional encoding.
        Args:
            query: Query tensor. (B, T_1, size)
            key: Key tensor. (B, T_2, size)
            value: Value tensor. (B, T_2, size)
            pos_enc: Positional embedding tensor. (B, 2 * T_1 - 1, size)
            mask: Source mask. (B, T_2)
            chunk_mask: Chunk mask. (T_1, T_1)
            left_context: Number of frames in left context.
        Returns:
            : Output tensor. (B, T_1, H * d_k)
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = self.compute_att_score(q, k, pos_enc, left_context=left_context)
        return self.forward_attention(v, scores, mask, chunk_mask=chunk_mask)
