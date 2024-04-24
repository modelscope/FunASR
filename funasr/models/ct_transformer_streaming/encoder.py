#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
from typing import List, Optional, Tuple

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.models.transformer.utils.repeat import repeat
from funasr.models.transformer.layer_norm import LayerNorm
from funasr.models.sanm.attention import MultiHeadedAttention
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.utils.subsampling import check_short_utt
from funasr.models.transformer.utils.subsampling import TooShortUttError
from funasr.models.transformer.embedding import SinusoidalPositionEncoder
from funasr.models.transformer.utils.multi_layer_conv import Conv1dLinear
from funasr.models.transformer.utils.mask import subsequent_mask, vad_mask
from funasr.models.transformer.utils.multi_layer_conv import MultiLayeredConv1d
from funasr.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from funasr.models.ct_transformer_streaming.attention import MultiHeadedAttentionSANMwithMask
from funasr.models.transformer.utils.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)


class EncoderLayerSANM(torch.nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayerSANM, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = torch.nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.in_size == self.size:
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            x = residual + attn
        else:
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)

        return x, cache


@tables.register("encoder_classes", "SANMVadEncoder")
class SANMVadEncoder(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group

    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        selfattention_layer_type: str = "sanm",
    ):
        super().__init__()
        self._output_size = output_size

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                SinusoidalPositionEncoder(),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        elif input_layer == "pe":
            self.embed = SinusoidalPositionEncoder()
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )

        elif selfattention_layer_type == "sanm":
            self.encoder_selfattn_layer = MultiHeadedAttentionSANMwithMask
            encoder_selfattn_layer_args0 = (
                attention_heads,
                input_size,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
            )

            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
            )

        self.encoders0 = repeat(
            1,
            lambda lnum: EncoderLayerSANM(
                input_size,
                output_size,
                self.encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.encoders = repeat(
            num_blocks - 1,
            lambda lnum: EncoderLayerSANM(
                output_size,
                output_size,
                self.encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        self.dropout = torch.nn.Dropout(dropout_rate)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        vad_indexes: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        sub_masks = subsequent_mask(masks.size(-1), device=xs_pad.device).unsqueeze(0)
        no_future_masks = masks & sub_masks
        xs_pad *= self.output_size() ** 0.5
        if self.embed is None:
            xs_pad = xs_pad
        elif (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        # xs_pad = self.dropout(xs_pad)
        mask_tup0 = [masks, no_future_masks]
        encoder_outs = self.encoders0(xs_pad, mask_tup0)
        xs_pad, _ = encoder_outs[0], encoder_outs[1]
        intermediate_outs = []

        for layer_idx, encoder_layer in enumerate(self.encoders):
            if layer_idx + 1 == len(self.encoders):
                # This is last layer.
                coner_mask = torch.ones(
                    masks.size(0),
                    masks.size(-1),
                    masks.size(-1),
                    device=xs_pad.device,
                    dtype=torch.bool,
                )
                for word_index, length in enumerate(ilens):
                    coner_mask[word_index, :, :] = vad_mask(
                        masks.size(-1), vad_indexes[word_index], device=xs_pad.device
                    )
                layer_mask = masks & coner_mask
            else:
                layer_mask = no_future_masks
            mask_tup1 = [masks, layer_mask]
            encoder_outs = encoder_layer(xs_pad, mask_tup1)
            xs_pad, layer_mask = encoder_outs[0], encoder_outs[1]

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None


class EncoderLayerSANMExport(torch.nn.Module):
    def __init__(
        self,
        model,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = model.self_attn
        self.feed_forward = model.feed_forward
        self.norm1 = model.norm1
        self.norm2 = model.norm2
        self.in_size = model.in_size
        self.size = model.size

    def forward(self, x, mask):

        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        if self.in_size == self.size:
            x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        return x, mask


@tables.register("encoder_classes", "SANMVadEncoderExport")
class SANMVadEncoderExport(torch.nn.Module):
    def __init__(
        self,
        model,
        max_seq_len=512,
        feats_dim=560,
        model_name="encoder",
        onnx: bool = True,
    ):
        super().__init__()
        self.embed = model.embed
        self.model = model
        self._output_size = model._output_size

        from funasr.utils.torch_function import sequence_mask

        self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

        from funasr.models.sanm.attention import MultiHeadedAttentionSANMExport

        if hasattr(model, "encoders0"):
            for i, d in enumerate(self.model.encoders0):
                if isinstance(d.self_attn, MultiHeadedAttentionSANMwithMask):
                    d.self_attn = MultiHeadedAttentionSANMExport(d.self_attn)
                self.model.encoders0[i] = EncoderLayerSANMExport(d)

        for i, d in enumerate(self.model.encoders):
            if isinstance(d.self_attn, MultiHeadedAttentionSANMwithMask):
                d.self_attn = MultiHeadedAttentionSANMExport(d.self_attn)
            self.model.encoders[i] = EncoderLayerSANMExport(d)

    def prepare_mask(self, mask, sub_masks):
        mask_3d_btd = mask[:, :, None]
        mask_4d_bhlt = (1 - sub_masks) * -10000.0

        return mask_3d_btd, mask_4d_bhlt

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        vad_masks: torch.Tensor,
        sub_masks: torch.Tensor,
    ):
        speech = speech * self._output_size**0.5
        mask = self.make_pad_mask(speech_lengths)
        vad_masks = self.prepare_mask(mask, vad_masks)
        mask = self.prepare_mask(mask, sub_masks)

        if self.embed is None:
            xs_pad = speech
        else:
            xs_pad = self.embed(speech)

        encoder_outs = self.model.encoders0(xs_pad, mask)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

        # encoder_outs = self.model.encoders(xs_pad, mask)
        for layer_idx, encoder_layer in enumerate(self.model.encoders):
            if layer_idx == len(self.model.encoders) - 1:
                mask = vad_masks
            encoder_outs = encoder_layer(xs_pad, mask)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.model.after_norm(xs_pad)

        return xs_pad, speech_lengths

    def get_output_size(self):
        return self.model.encoders[0].size
