"""Conformer block for Transducer encoder."""

from typing import Dict, Optional, Tuple

import torch


class Conformer(torch.nn.Module):
    """Conformer module definition.

    Args:
        block_size: Input/output size.
        self_att: Self-attention module instance.
        feed_forward: Feed-forward module instance.
        feed_forward_macaron: Feed-forward module instance for macaron network.
        conv_mod: Convolution module instance.
        norm_class: Normalization module class.
        norm_args: Normalization module arguments.
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        block_size: int,
        self_att: torch.nn.Module,
        feed_forward: torch.nn.Module,
        feed_forward_macaron: torch.nn.Module,
        conv_mod: torch.nn.Module,
        norm_class: torch.nn.Module = torch.nn.LayerNorm,
        norm_args: Dict = {},
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a Conformer object."""
        super().__init__()

        self.self_att = self_att

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.feed_forward_scale = 0.5

        self.conv_mod = conv_mod

        self.norm_feed_forward = norm_class(block_size, **norm_args)
        self.norm_self_att = norm_class(block_size, **norm_args)

        self.norm_macaron = norm_class(block_size, **norm_args)
        self.norm_conv = norm_class(block_size, **norm_args)
        self.norm_final = norm_class(block_size, **norm_args)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.block_size = block_size
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
                    self.block_size,
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
            x: Conformer input sequences. (B, T, D_block)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)
            mask: Source mask. (B, T)
            chunk_mask: Chunk mask. (T_2, T_2)

        Returns:
            x: Conformer output sequences. (B, T, D_block)
            mask: Source mask. (B, T)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)

        """
        residual = x

        x = self.norm_macaron(x)
        x = residual + self.feed_forward_scale * self.dropout(
            self.feed_forward_macaron(x)
        )

        residual = x
        x = self.norm_self_att(x)
        x_q = x
        x = residual + self.dropout(
            self.self_att(
                x_q,
                x,
                x,
                pos_enc,
                mask,
                chunk_mask=chunk_mask,
            )
        )

        residual = x

        x = self.norm_conv(x)
        x, _ = self.conv_mod(x)
        x = residual + self.dropout(x)
        residual = x

        x = self.norm_feed_forward(x)
        x = residual + self.feed_forward_scale * self.dropout(self.feed_forward(x))

        x = self.norm_final(x)
        return x, mask, pos_enc

    def chunk_forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int = 16,
        left_context: int = 0,
        right_context: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode chunk of input sequence.

        Args:
            x: Conformer input sequences. (B, T, D_block)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)
            mask: Source mask. (B, T_2)
            left_context: Number of frames in left context.
            right_context: Number of frames in right context.

        Returns:
            x: Conformer output sequences. (B, T, D_block)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)

        """
        residual = x

        x = self.norm_macaron(x)
        x = residual + self.feed_forward_scale * self.feed_forward_macaron(x)

        residual = x
        x = self.norm_self_att(x)
        if left_context > 0:
            key = torch.cat([self.cache[0], x], dim=1)
        else:
            key = x
        val = key

        if right_context > 0:
            att_cache = key[:, -(left_context + right_context) : -right_context, :]
        else:
            att_cache = key[:, -left_context:, :]
        x = residual + self.self_att(
            x,
            key,
            val,
            pos_enc,
            mask,
            left_context=left_context,
        )

        residual = x
        x = self.norm_conv(x)
        x, conv_cache = self.conv_mod(
            x, cache=self.cache[1], right_context=right_context
        )
        x = residual + x
        residual = x
        
        x = self.norm_feed_forward(x)
        x = residual + self.feed_forward_scale * self.feed_forward(x)

        x = self.norm_final(x)
        self.cache = [att_cache, conv_cache]
       
        return x, pos_enc
