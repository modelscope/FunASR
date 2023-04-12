"""Branchformer block for Transducer encoder."""

from typing import Dict, Optional, Tuple

import torch


class Branchformer(torch.nn.Module):
    """Branchformer module definition.

    Reference: https://arxiv.org/pdf/2207.02971.pdf

    Args:
        block_size: Input/output size.
        linear_size: Linear layers' hidden size.
        self_att: Self-attention module instance.
        conv_mod: Convolution module instance.
        norm_class: Normalization class.
        norm_args: Normalization module arguments.
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        block_size: int,
        linear_size: int,
        self_att: torch.nn.Module,
        conv_mod: torch.nn.Module,
        norm_class: torch.nn.Module = torch.nn.LayerNorm,
        norm_args: Dict = {},
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a Branchformer object."""
        super().__init__()

        self.self_att = self_att
        self.conv_mod = conv_mod

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(block_size, linear_size), torch.nn.GELU()
        )
        self.channel_proj2 = torch.nn.Linear(linear_size // 2, block_size)

        self.merge_proj = torch.nn.Linear(block_size + block_size, block_size)

        self.norm_self_att = norm_class(block_size, **norm_args)
        self.norm_mlp = norm_class(block_size, **norm_args)
        self.norm_final = norm_class(block_size, **norm_args)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.block_size = block_size
        self.linear_size = linear_size
        self.cache = None

    def reset_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """Initialize/Reset self-attention and convolution modules cache for streaming.

        Args:
            left_context: Number of left frames during chunk-by-chunk inference.
            device: Device to use for cache tensor.

        """
        self.cache = [
            torch.zeros(
                (1, left_context, self.block_size),
                device=device,
            ),
            torch.zeros(
                (
                    1,
                    self.linear_size // 2,
                    self.conv_mod.kernel_size - 1,
                ),
                device=device,
            ),
        ]

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input sequences.

        Args:
            x: Branchformer input sequences. (B, T, D_block)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)
            mask: Source mask. (B, T)
            chunk_mask: Chunk mask. (T_2, T_2)

        Returns:
            x: Branchformer output sequences. (B, T, D_block)
            mask: Source mask. (B, T)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)

        """
        x1 = x
        x2 = x

        x1 = self.norm_self_att(x1)

        x1 = self.dropout(
            self.self_att(x1, x1, x1, pos_enc, mask=mask, chunk_mask=chunk_mask)
        )

        x2 = self.norm_mlp(x2)

        x2 = self.channel_proj1(x2)
        x2, _ = self.conv_mod(x2)
        x2 = self.channel_proj2(x2)

        x2 = self.dropout(x2)

        x = x + self.dropout(self.merge_proj(torch.cat([x1, x2], dim=-1)))

        x = self.norm_final(x)

        return x, mask, pos_enc

    def chunk_forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        left_context: int = 0,
        right_context: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode chunk of input sequence.

        Args:
            x: Branchformer input sequences. (B, T, D_block)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)
            mask: Source mask. (B, T_2)
            left_context: Number of frames in left context.
            right_context: Number of frames in right context.

        Returns:
            x: Branchformer output sequences. (B, T, D_block)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)

        """
        x1 = x
        x2 = x

        x1 = self.norm_self_att(x1)

        if left_context > 0:
            key = torch.cat([self.cache[0], x1], dim=1)
        else:
            key = x1
        val = key

        if right_context > 0:
            att_cache = key[:, -(left_context + right_context) : -right_context, :]
        else:
            att_cache = key[:, -left_context:, :]

        x1 = self.self_att(x1, key, val, pos_enc, mask=mask, left_context=left_context)

        x2 = self.norm_mlp(x2)
        x2 = self.channel_proj1(x2)

        x2, conv_cache = self.conv_mod(
            x2, cache=self.cache[1], right_context=right_context
        )

        x2 = self.channel_proj2(x2)

        x = x + self.merge_proj(torch.cat([x1, x2], dim=-1))

        x = self.norm_final(x)
        self.cache = [att_cache, conv_cache]

        return x, pos_enc
