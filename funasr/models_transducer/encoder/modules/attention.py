"""Multi-Head attention layers with relative positional encoding."""

import math
from typing import Optional, Tuple

import torch


class RelPositionMultiHeadedAttention(torch.nn.Module):
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

        q = (
            self.linear_q(query)
            .view(n_batch, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.linear_k(key)
            .view(n_batch, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.linear_v(value)
            .view(n_batch, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

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
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
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
