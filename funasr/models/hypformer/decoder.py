# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple

import torch
from torch import nn


from funasr.models.transformer.attention import MultiHeadedAttention
from funasr.models.transformer.utils.dynamic_conv import DynamicConvolution
from funasr.models.transformer.utils.dynamic_conv2d import DynamicConvolution2D
from funasr.models.transformer.embedding import PositionalEncoding
from funasr.models.transformer.layer_norm import LayerNorm
from funasr.models.transformer.utils.lightconv import LightweightConvolution
from funasr.models.transformer.utils.lightconv2d import LightweightConvolution2D
from funasr.models.transformer.utils.mask import subsequent_mask
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from funasr.models.transformer.utils.repeat import repeat
from funasr.models.transformer.scorers.scorer_interface import BatchScorerInterface

from funasr.register import tables

class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)


    """

    def __init__(
            self,
            size,
            self_attn,
            src_attn,
            feed_forward,
            dropout_rate,
            normalize_before=True,
            concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask

@tables.register("decoder_classes", "HypformerDecoder")
class HypformerDecoder(nn.Module, BatchScorerInterface):
    def __init__(
            self,
            vocab_size: int,
            encoder_output_size: int,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            self_attention_dropout_rate: float = 0.0,
            src_attention_dropout_rate: float = 0.0,
            input_layer: str = "embed",
            use_output_layer: bool = True,
            pos_enc_class=PositionalEncoding,
            normalize_before: bool = True,
            concat_after: bool = False,
    ):
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        attention_dim = encoder_output_size
        self.decoders = repeat(num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None


    def forward(
            self,
            hs_pad: torch.Tensor,
            hlens: torch.Tensor,
            ys_in_pad: torch.Tensor,
            ys_in_lens: torch.Tensor,
            hyp_in_pad: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, T = hyp_in_pad.size()
        device = hyp_in_pad.device

        ys_in_pad_matrix = ys_in_pad.unsqueeze(1).expand(-1, T, -1)
        hpy_mask = torch.tril(torch.ones(T, T), diagonal=0).to(device)
        hpy_mask = hpy_mask.unsqueeze(0).expand(B, -1, -1).bool()
        ys_in_pad_matrix = ys_in_pad_matrix * hpy_mask

        hyp_in_pad_matrix = hyp_in_pad.unsqueeze(1).expand(-1, T, -1)
        hyp_in_pad_matrix = hyp_in_pad_matrix * ~hpy_mask

        teach_force = ys_in_pad_matrix + hyp_in_pad_matrix
        teach_force = teach_force.reshape(-1, T)
        teach_force_lens = ys_in_lens.reshape(-1, 1).repeat(1, T).reshape(-1)

        memory = hs_pad
        memory = memory.unsqueeze(1).expand(-1, T, -1, -1).reshape(-1, memory.size(-2), memory.size(-1))
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(memory.device)
        memory_mask = memory_mask.unsqueeze(1).expand(-1, T, -1, -1).reshape(-1, memory_mask.size(-2), memory_mask.size(-1))

        pickup_mask = torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1).unsqueeze(-1)

        tgt = teach_force
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(teach_force_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = torch.ones(1,  tgt_mask.size(-1), tgt_mask.size(-1), device=tgt_mask.device).bool()
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        # Padding for Longformer
        if memory_mask.shape[-1] != memory.shape[1]:
            padlen = memory.shape[1] - memory_mask.shape[-1]
            memory_mask = torch.nn.functional.pad(memory_mask, (0, padlen), "constant", False)

        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)

        tgt_mask  = tgt_mask[0::T, :, :]
        olens = tgt_mask.sum(1)

        x = x.reshape(B, T, x.size(-2), x.size(-1))
        x = x * pickup_mask
        x = x.sum(dim=2)

        return x, olens

    
    def forward_one_step(
            self,
            memory: torch.Tensor,  # [batch, maxlen_in, feat]
            memory_mask: torch.Tensor,  # [batch]
            hyp_in_pad: torch.Tensor,  # [batch, T]
            hyps_mask: torch.Tensor,  # [batch]
            decoding_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, T = hyp_in_pad.size()  # [batch, T]
        device = hyp_in_pad.device

        tgt = hyp_in_pad  # [batch, T]
        tgt_mask = hyps_mask # [batch, 1, T]
        m = torch.ones(1,  tgt_mask.size(-1), tgt_mask.size(-1), device=tgt_mask.device).bool() 
        tgt_mask = tgt_mask & m  # [batch, T, T]

        # Padding for Longformer
        if memory_mask.shape[-1] != memory.shape[1]:
            padlen = memory.shape[1] - memory_mask.shape[-1]
            memory_mask = torch.nn.functional.pad(memory_mask, (0, padlen), "constant", False)

        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)

        x = x.reshape(B, x.size(-2), x.size(-1))  # [batch, maxlen_out, feat]
        x = x[:, decoding_idx]  # [batch, feat]
        return x, None
    

    def forward_hpy_step(
            self,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            decoding_idx: int,
            cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(x, tgt_mask, memory, memory_mask, cache=c)
            new_cache.append(x)

        x = x[:, decoding_idx]

        if self.normalize_before:
            y = self.after_norm(x)
        else:
            y = x
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache


    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
            self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
