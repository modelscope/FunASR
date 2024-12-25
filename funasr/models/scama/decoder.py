#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from typing import List
from typing import Tuple
import logging
import torch
import torch.nn as nn
import numpy as np

from funasr.models.scama import utils as myutils
from funasr.models.transformer.decoder import BaseTransformerDecoder

from funasr.models.sanm.attention import (
    MultiHeadedAttentionSANMDecoder,
    MultiHeadedAttentionCrossAtt,
)
from funasr.models.transformer.embedding import PositionalEncoding
from funasr.models.transformer.layer_norm import LayerNorm
from funasr.models.sanm.positionwise_feed_forward import PositionwiseFeedForwardDecoderSANM
from funasr.models.transformer.utils.repeat import repeat

from funasr.register import tables


class DecoderLayerSANM(nn.Module):
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
        super(DecoderLayerSANM, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        if self_attn is not None:
            self.norm2 = LayerNorm(size)
        if src_attn is not None:
            self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
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
        # tgt = self.dropout(tgt)
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn:
            if self.normalize_before:
                tgt = self.norm2(tgt)
            x, _ = self.self_attn(tgt, tgt_mask)
            x = residual + self.dropout(x)

        if self.src_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm3(x)

            x = residual + self.dropout(self.src_attn(x, memory, memory_mask))

        return x, tgt_mask, memory, memory_mask, cache

    def forward_one_step(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
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
        # tgt = self.dropout(tgt)
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn:
            if self.normalize_before:
                tgt = self.norm2(tgt)
            if self.training:
                cache = None
            x, cache = self.self_attn(tgt, tgt_mask, cache=cache)
            x = residual + self.dropout(x)

        if self.src_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm3(x)

            x = residual + self.dropout(self.src_attn(x, memory, memory_mask))

        return x, tgt_mask, memory, memory_mask, cache

    def forward_chunk(
        self, tgt, memory, fsmn_cache=None, opt_cache=None, chunk_size=None, look_back=0
    ):
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
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn:
            if self.normalize_before:
                tgt = self.norm2(tgt)
            x, fsmn_cache = self.self_attn(tgt, None, fsmn_cache)
            x = residual + self.dropout(x)

        if self.src_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm3(x)

            x, opt_cache = self.src_attn.forward_chunk(x, memory, opt_cache, chunk_size, look_back)
            x = residual + x

        return x, memory, fsmn_cache, opt_cache


@tables.register("decoder_classes", "FsmnDecoderSCAMAOpt")
class FsmnDecoderSCAMAOpt(BaseTransformerDecoder):
    """
    Author: Shiliang Zhang, Zhifu Gao, Haoneng Luo, Ming Lei, Jie Gao, Zhijie Yan, Lei Xie
    SCAMA: Streaming chunk-aware multihead attention for online end-to-end speech recognition
    https://arxiv.org/abs/2006.01712
    """

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
        att_layer_num: int = 6,
        kernel_size: int = 21,
        sanm_shfit: int = None,
        concat_embeds: bool = False,
        attention_dim: int = None,
        tf2torch_tensor_name_prefix_torch: str = "decoder",
        tf2torch_tensor_name_prefix_tf: str = "seq2seq/decoder",
        embed_tensor_name_prefix_tf: str = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )
        if attention_dim is None:
            attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
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

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        self.att_layer_num = att_layer_num
        self.num_blocks = num_blocks
        if sanm_shfit is None:
            sanm_shfit = (kernel_size - 1) // 2
        self.decoders = repeat(
            att_layer_num,
            lambda lnum: DecoderLayerSANM(
                attention_dim,
                MultiHeadedAttentionSANMDecoder(
                    attention_dim, self_attention_dropout_rate, kernel_size, sanm_shfit=sanm_shfit
                ),
                MultiHeadedAttentionCrossAtt(
                    attention_heads,
                    attention_dim,
                    src_attention_dropout_rate,
                    encoder_output_size=encoder_output_size,
                ),
                PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if num_blocks - att_layer_num <= 0:
            self.decoders2 = None
        else:
            self.decoders2 = repeat(
                num_blocks - att_layer_num,
                lambda lnum: DecoderLayerSANM(
                    attention_dim,
                    MultiHeadedAttentionSANMDecoder(
                        attention_dim,
                        self_attention_dropout_rate,
                        kernel_size,
                        sanm_shfit=sanm_shfit,
                    ),
                    None,
                    PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )

        self.decoders3 = repeat(
            1,
            lambda lnum: DecoderLayerSANM(
                attention_dim,
                None,
                None,
                PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if concat_embeds:
            self.embed_concat_ffn = repeat(
                1,
                lambda lnum: DecoderLayerSANM(
                    attention_dim + encoder_output_size,
                    None,
                    None,
                    PositionwiseFeedForwardDecoderSANM(
                        attention_dim + encoder_output_size,
                        linear_units,
                        dropout_rate,
                        adim=attention_dim,
                    ),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        else:
            self.embed_concat_ffn = None
        self.concat_embeds = concat_embeds
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf
        self.embed_tensor_name_prefix_tf = embed_tensor_name_prefix_tf

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        chunk_mask: torch.Tensor = None,
        pre_acoustic_embeds: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        tgt_mask = myutils.sequence_mask(ys_in_lens, device=tgt.device)[:, :, None]

        memory = hs_pad
        memory_mask = myutils.sequence_mask(hlens, device=memory.device)[:, None, :]
        if chunk_mask is not None:
            memory_mask = memory_mask * chunk_mask
            if tgt_mask.size(1) != memory_mask.size(1):
                memory_mask = torch.cat((memory_mask, memory_mask[:, -2:-1, :]), dim=1)

        x = self.embed(tgt)

        if pre_acoustic_embeds is not None and self.concat_embeds:
            x = torch.cat((x, pre_acoustic_embeds), dim=-1)
            x, _, _, _, _ = self.embed_concat_ffn(x, None, None, None, None)

        x, tgt_mask, memory, memory_mask, _ = self.decoders(x, tgt_mask, memory, memory_mask)
        if self.decoders2 is not None:
            x, tgt_mask, memory, memory_mask, _ = self.decoders2(x, tgt_mask, memory, memory_mask)
        x, tgt_mask, memory, memory_mask, _ = self.decoders3(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        return x, olens

    def score(
        self,
        ys,
        state,
        x,
        x_mask=None,
        pre_acoustic_embeds: torch.Tensor = None,
    ):
        """Score."""
        ys_mask = myutils.sequence_mask(
            torch.tensor([len(ys)], dtype=torch.int32), device=x.device
        )[:, :, None]
        logp, state = self.forward_one_step(
            ys.unsqueeze(0),
            ys_mask,
            x.unsqueeze(0),
            memory_mask=x_mask,
            pre_acoustic_embeds=pre_acoustic_embeds,
            cache=state,
        )
        return logp.squeeze(0), state

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor = None,
        pre_acoustic_embeds: torch.Tensor = None,
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

        x = tgt[:, -1:]
        tgt_mask = None
        x = self.embed(x)

        if pre_acoustic_embeds is not None and self.concat_embeds:
            x = torch.cat((x, pre_acoustic_embeds), dim=-1)
            x, _, _, _, _ = self.embed_concat_ffn(x, None, None, None, None)

        if cache is None:
            cache_layer_num = len(self.decoders)
            if self.decoders2 is not None:
                cache_layer_num += len(self.decoders2)
            cache = [None] * cache_layer_num
        new_cache = []
        # for c, decoder in zip(cache, self.decoders):
        for i in range(self.att_layer_num):
            decoder = self.decoders[i]
            c = cache[i]
            x, tgt_mask, memory, memory_mask, c_ret = decoder.forward_one_step(
                x, tgt_mask, memory, memory_mask, cache=c
            )
            new_cache.append(c_ret)

        if self.num_blocks - self.att_layer_num >= 1:
            for i in range(self.num_blocks - self.att_layer_num):
                j = i + self.att_layer_num
                decoder = self.decoders2[i]
                c = cache[j]
                x, tgt_mask, memory, memory_mask, c_ret = decoder.forward_one_step(
                    x, tgt_mask, memory, memory_mask, cache=c
                )
                new_cache.append(c_ret)

        for decoder in self.decoders3:
            x, tgt_mask, memory, memory_mask, _ = decoder.forward_one_step(
                x, tgt_mask, memory, None, cache=None
            )

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = self.output_layer(y)
            y = torch.log_softmax(y, dim=-1)

        return y, new_cache
