#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import logging
import numpy as np
from typing import Tuple

from funasr.register import tables
from funasr.models.scama import utils as myutils
from funasr.models.transformer.utils.repeat import repeat
from funasr.models.transformer.layer_norm import LayerNorm
from funasr.models.transformer.embedding import PositionalEncoding
from funasr.models.paraformer.decoder import DecoderLayerSANM, ParaformerSANMDecoder
from funasr.models.sanm.positionwise_feed_forward import PositionwiseFeedForwardDecoderSANM
from funasr.models.sanm.attention import (
    MultiHeadedAttentionSANMDecoder,
    MultiHeadedAttentionCrossAtt,
)


class ContextualDecoderLayer(torch.nn.Module):
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
        super(ContextualDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        if self_attn is not None:
            self.norm2 = LayerNorm(size)
        if src_attn is not None:
            self.norm3 = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = torch.nn.Linear(size + size, size)
            self.concat_linear2 = torch.nn.Linear(size + size, size)

    def forward(
        self,
        tgt,
        tgt_mask,
        memory,
        memory_mask,
        cache=None,
    ):
        # tgt = self.dropout(tgt)
        if isinstance(tgt, Tuple):
            tgt, _ = tgt
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if self.training:
            cache = None
        x, cache = self.self_attn(tgt, tgt_mask, cache=cache)
        x = residual + self.dropout(x)
        x_self_attn = x

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = self.src_attn(x, memory, memory_mask)
        x_src_attn = x

        x = residual + self.dropout(x)
        return x, tgt_mask, x_self_attn, x_src_attn


class ContextualBiasDecoder(torch.nn.Module):
    def __init__(
        self,
        size,
        src_attn,
        dropout_rate,
        normalize_before=True,
    ):
        """Construct an DecoderLayer object."""
        super(ContextualBiasDecoder, self).__init__()
        self.size = size
        self.src_attn = src_attn
        if src_attn is not None:
            self.norm3 = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        x = tgt
        if self.src_attn is not None:
            if self.normalize_before:
                x = self.norm3(x)
            x = self.dropout(self.src_attn(x, memory, memory_mask))
        return x, tgt_mask, memory, memory_mask, cache


@tables.register("decoder_classes", "ContextualParaformerDecoder")
class ContextualParaformerDecoder(ParaformerSANMDecoder):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2006.01713
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
        sanm_shfit: int = 0,
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

        attention_dim = encoder_output_size
        if input_layer == "none":
            self.embed = None
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                # pos_enc_class(attention_dim, positional_dropout_rate),
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
            att_layer_num - 1,
            lambda lnum: DecoderLayerSANM(
                attention_dim,
                MultiHeadedAttentionSANMDecoder(
                    attention_dim, self_attention_dropout_rate, kernel_size, sanm_shfit=sanm_shfit
                ),
                MultiHeadedAttentionCrossAtt(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.bias_decoder = ContextualBiasDecoder(
            size=attention_dim,
            src_attn=MultiHeadedAttentionCrossAtt(
                attention_heads, attention_dim, src_attention_dropout_rate
            ),
            dropout_rate=dropout_rate,
            normalize_before=True,
        )
        self.bias_output = torch.nn.Conv1d(attention_dim * 2, attention_dim, 1, bias=False)
        self.last_decoder = ContextualDecoderLayer(
            attention_dim,
            MultiHeadedAttentionSANMDecoder(
                attention_dim, self_attention_dropout_rate, kernel_size, sanm_shfit=sanm_shfit
            ),
            MultiHeadedAttentionCrossAtt(
                attention_heads, attention_dim, src_attention_dropout_rate
            ),
            PositionwiseFeedForwardDecoderSANM(attention_dim, linear_units, dropout_rate),
            dropout_rate,
            normalize_before,
            concat_after,
        )
        if num_blocks - att_layer_num <= 0:
            self.decoders2 = None
        else:
            self.decoders2 = repeat(
                num_blocks - att_layer_num,
                lambda lnum: DecoderLayerSANM(
                    attention_dim,
                    MultiHeadedAttentionSANMDecoder(
                        attention_dim, self_attention_dropout_rate, kernel_size, sanm_shfit=0
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

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        contextual_info: torch.Tensor,
        clas_scale: float = 1.0,
        return_hidden: bool = False,
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

        x = tgt
        x, tgt_mask, memory, memory_mask, _ = self.decoders(x, tgt_mask, memory, memory_mask)
        _, _, x_self_attn, x_src_attn = self.last_decoder(x, tgt_mask, memory, memory_mask)

        # contextual paraformer related
        contextual_length = torch.Tensor([contextual_info.shape[1]]).int().repeat(hs_pad.shape[0])
        contextual_mask = myutils.sequence_mask(contextual_length, device=memory.device)[:, None, :]
        cx, tgt_mask, _, _, _ = self.bias_decoder(
            x_self_attn, tgt_mask, contextual_info, memory_mask=contextual_mask
        )

        if self.bias_output is not None:
            x = torch.cat([x_src_attn, cx * clas_scale], dim=2)
            x = self.bias_output(x.transpose(1, 2)).transpose(1, 2)  # 2D -> D
            x = x_self_attn + self.dropout(x)

        if self.decoders2 is not None:
            x, tgt_mask, memory, memory_mask, _ = self.decoders2(x, tgt_mask, memory, memory_mask)

        x, tgt_mask, memory, memory_mask, _ = self.decoders3(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        olens = tgt_mask.sum(1)
        if self.output_layer is not None and return_hidden is False:
            x = self.output_layer(x)
        return x, olens


@tables.register("decoder_classes", "ContextualParaformerDecoderExport")
class ContextualParaformerDecoderExport(torch.nn.Module):
    def __init__(
        self,
        model,
        max_seq_len=512,
        model_name="decoder",
        onnx: bool = True,
        **kwargs,
    ):
        super().__init__()
        from funasr.utils.torch_function import sequence_mask

        self.model = model
        self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

        from funasr.models.sanm.attention import MultiHeadedAttentionSANMDecoderExport
        from funasr.models.sanm.attention import MultiHeadedAttentionCrossAttExport
        from funasr.models.paraformer.decoder import DecoderLayerSANMExport
        from funasr.models.transformer.positionwise_feed_forward import (
            PositionwiseFeedForwardDecoderSANMExport,
        )

        for i, d in enumerate(self.model.decoders):
            if isinstance(d.feed_forward, PositionwiseFeedForwardDecoderSANM):
                d.feed_forward = PositionwiseFeedForwardDecoderSANMExport(d.feed_forward)
            if isinstance(d.self_attn, MultiHeadedAttentionSANMDecoder):
                d.self_attn = MultiHeadedAttentionSANMDecoderExport(d.self_attn)
            if isinstance(d.src_attn, MultiHeadedAttentionCrossAtt):
                d.src_attn = MultiHeadedAttentionCrossAttExport(d.src_attn)
            self.model.decoders[i] = DecoderLayerSANMExport(d)

        if self.model.decoders2 is not None:
            for i, d in enumerate(self.model.decoders2):
                if isinstance(d.feed_forward, PositionwiseFeedForwardDecoderSANM):
                    d.feed_forward = PositionwiseFeedForwardDecoderSANMExport(d.feed_forward)
                if isinstance(d.self_attn, MultiHeadedAttentionSANMDecoder):
                    d.self_attn = MultiHeadedAttentionSANMDecoderExport(d.self_attn)
                self.model.decoders2[i] = DecoderLayerSANMExport(d)

        for i, d in enumerate(self.model.decoders3):
            if isinstance(d.feed_forward, PositionwiseFeedForwardDecoderSANM):
                d.feed_forward = PositionwiseFeedForwardDecoderSANMExport(d.feed_forward)
            self.model.decoders3[i] = DecoderLayerSANMExport(d)

        self.output_layer = model.output_layer
        self.after_norm = model.after_norm
        self.model_name = model_name

        # bias decoder
        if isinstance(self.model.bias_decoder.src_attn, MultiHeadedAttentionCrossAtt):
            self.model.bias_decoder.src_attn = MultiHeadedAttentionCrossAttExport(
                self.model.bias_decoder.src_attn
            )
        self.bias_decoder = self.model.bias_decoder

        # last decoder
        if isinstance(self.model.last_decoder.src_attn, MultiHeadedAttentionCrossAtt):
            self.model.last_decoder.src_attn = MultiHeadedAttentionCrossAttExport(
                self.model.last_decoder.src_attn
            )
        if isinstance(self.model.last_decoder.self_attn, MultiHeadedAttentionSANMDecoder):
            self.model.last_decoder.self_attn = MultiHeadedAttentionSANMDecoderExport(
                self.model.last_decoder.self_attn
            )
        if isinstance(self.model.last_decoder.feed_forward, PositionwiseFeedForwardDecoderSANM):
            self.model.last_decoder.feed_forward = PositionwiseFeedForwardDecoderSANMExport(
                self.model.last_decoder.feed_forward
            )
        self.last_decoder = self.model.last_decoder
        self.bias_output = self.model.bias_output
        self.dropout = self.model.dropout

    def prepare_mask(self, mask):
        mask_3d_btd = mask[:, :, None]
        if len(mask.shape) == 2:
            mask_4d_bhlt = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask_4d_bhlt = 1 - mask[:, None, :]
        mask_4d_bhlt = mask_4d_bhlt * -10000.0

        return mask_3d_btd, mask_4d_bhlt

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        bias_embed: torch.Tensor,
    ):

        tgt = ys_in_pad
        tgt_mask = self.make_pad_mask(ys_in_lens)
        tgt_mask, _ = self.prepare_mask(tgt_mask)
        # tgt_mask = myutils.sequence_mask(ys_in_lens, device=tgt.device)[:, :, None]

        memory = hs_pad
        memory_mask = self.make_pad_mask(hlens)
        _, memory_mask = self.prepare_mask(memory_mask)
        # memory_mask = myutils.sequence_mask(hlens, device=memory.device)[:, None, :]

        x = tgt
        x, tgt_mask, memory, memory_mask, _ = self.model.decoders(x, tgt_mask, memory, memory_mask)

        _, _, x_self_attn, x_src_attn = self.last_decoder(x, tgt_mask, memory, memory_mask)

        # contextual paraformer related
        contextual_length = torch.Tensor([bias_embed.shape[1]]).int().repeat(hs_pad.shape[0])
        # contextual_mask = myutils.sequence_mask(contextual_length, device=memory.device)[:, None, :]
        contextual_mask = self.make_pad_mask(contextual_length)
        contextual_mask, _ = self.prepare_mask(contextual_mask)
        # import pdb; pdb.set_trace()
        contextual_mask = contextual_mask.transpose(2, 1).unsqueeze(1)
        cx, tgt_mask, _, _, _ = self.bias_decoder(
            x_self_attn, tgt_mask, bias_embed, memory_mask=contextual_mask
        )

        if self.bias_output is not None:
            x = torch.cat([x_src_attn, cx], dim=2)
            x = self.bias_output(x.transpose(1, 2)).transpose(1, 2)  # 2D -> D
            x = x_self_attn + self.dropout(x)

        if self.model.decoders2 is not None:
            x, tgt_mask, memory, memory_mask, _ = self.model.decoders2(
                x, tgt_mask, memory, memory_mask
            )
        x, tgt_mask, memory, memory_mask, _ = self.model.decoders3(x, tgt_mask, memory, memory_mask)
        x = self.after_norm(x)
        x = self.output_layer(x)

        return x, ys_in_lens
