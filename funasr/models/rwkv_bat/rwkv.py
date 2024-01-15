"""Receptance Weighted Key Value (RWKV) block definition.

Based/modified from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model.py

"""

from typing import Dict, Optional, Tuple

import torch

from funasr.models.rwkv_bat.rwkv_attention import EncoderSelfAttention, DecoderSelfAttention
from funasr.models.rwkv_bat.rwkv_feed_forward import FeedForward
from funasr.models.transformer.layer_norm import LayerNorm

class RWKV(torch.nn.Module):
    """RWKV module.

    Args:
        size: Input/Output size.
        linear_size: Feed-forward hidden size.
        attention_size: SelfAttention hidden size.
        context_size: Context size for WKV computation.
        block_id: Block index.
        num_blocks: Number of blocks in the architecture.
        normalization_class: Normalization layer class.
        normalization_args: Normalization layer arguments.
        att_dropout_rate: Dropout rate for the attention module.
        ffn_dropout_rate: Dropout rate for the feed-forward module.

    """

    def __init__(
        self,
        size: int,
        linear_size: int,
        attention_size: int,
        context_size: int,
        block_id: int,
        num_blocks: int,
        att_dropout_rate: float = 0.0,
        ffn_dropout_rate: float = 0.0,
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a RWKV object."""
        super().__init__()

        self.layer_norm_att = LayerNorm(size)
        self.layer_norm_ffn = LayerNorm(size)

        self.att = EncoderSelfAttention(
            size, attention_size, context_size, block_id, att_dropout_rate, num_blocks
        )
        self.dropout_att = torch.nn.Dropout(p=dropout_rate)

        self.ffn = FeedForward(size, linear_size, block_id, ffn_dropout_rate, num_blocks)
        self.dropout_ffn = torch.nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute receptance weighted key value.

        Args:
            x: RWKV input sequences. (B, L, size)
            state: Decoder hidden states. [5 x (B, D_att/size, N)]

        Returns:
            x: RWKV output sequences. (B, L, size)
            x: Decoder hidden states. [5 x (B, D_att/size, N)]

        """
        att, state = self.att(self.layer_norm_att(x), state=state)
        x = x + self.dropout_att(att)
        ffn, state = self.ffn(self.layer_norm_ffn(x), state=state)
        x = x + self.dropout_ffn(ffn)
        return x, state

class RWKVDecoderLayer(torch.nn.Module):
    """RWKV module.

    Args:
        size: Input/Output size.
        linear_size: Feed-forward hidden size.
        attention_size: SelfAttention hidden size.
        context_size: Context size for WKV computation.
        block_id: Block index.
        num_blocks: Number of blocks in the architecture.
        normalization_class: Normalization layer class.
        normalization_args: Normalization layer arguments.
        att_dropout_rate: Dropout rate for the attention module.
        ffn_dropout_rate: Dropout rate for the feed-forward module.

    """

    def __init__(
        self,
        size: int,
        linear_size: int,
        attention_size: int,
        context_size: int,
        block_id: int,
        num_blocks: int,
        att_dropout_rate: float = 0.0,
        ffn_dropout_rate: float = 0.0,
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a RWKV object."""
        super().__init__()

        self.layer_norm_att = LayerNorm(size)
        self.layer_norm_ffn = LayerNorm(size)

        self.att = DecoderSelfAttention(
            size, attention_size, context_size, block_id, att_dropout_rate, num_blocks
        )
        self.dropout_att = torch.nn.Dropout(p=dropout_rate)

        self.ffn = FeedForward(size, linear_size, block_id, ffn_dropout_rate, num_blocks)
        self.dropout_ffn = torch.nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute receptance weighted key value.

        Args:
            x: RWKV input sequences. (B, L, size)
            state: Decoder hidden states. [5 x (B, D_att/size, N)]

        Returns:
            x: RWKV output sequences. (B, L, size)
            x: Decoder hidden states. [5 x (B, D_att/size, N)]

        """
        att, state = self.att(self.layer_norm_att(x), state=state)
        x = x + self.dropout_att(att)

        ffn, state = self.ffn(self.layer_norm_ffn(x), state=state)
        x = x + self.dropout_ffn(ffn)

        return x, state
