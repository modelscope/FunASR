import logging
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import torch
from torch import nn
from funasr.models.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from funasr.models.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from funasr.models.transformer.layer_norm import LayerNorm
from funasr.models.transformer.utils.multi_layer_conv import Conv1dLinear
from funasr.models.transformer.utils.multi_layer_conv import MultiLayeredConv1d
from funasr.models.transformer.utils.nets_utils import get_activation
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.utils.mask import subsequent_mask, causal_block_mask
from funasr.models.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from funasr.models.transformer.utils.repeat import repeat
from funasr.models.transformer.utils.subsampling import (
    Conv2dSubsampling, Conv2dSubsampling2, Conv2dSubsampling6, Conv2dSubsampling8, TooShortUttError,
    check_short_utt, Conv2dSubsamplingPad
)
import torch.nn.functional as F
from funasr.models.llm_asr.conformer_encoder import ConvolutionModule, EncoderLayer
from funasr.models.ctc.ctc import CTC


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=True,
                 out_channels=None, name="conv", channel_first=True, stride=2, causal=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.channel_first = channel_first
        self.stride = stride
        self.causal = causal

        self.conv = None
        if use_conv_transpose:
            # transpose conv doesn't support causal mode.
            assert not causal
            kernel_size = stride*2 + stride % 2
            padding = (kernel_size - stride) // 2
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, kernel_size, stride, padding)
        elif use_conv:
            # In this mode, first repeat interpolate, than conv with stride=1
            self.conv = nn.Conv1d(
                self.channels, self.out_channels, stride*2+1, stride=1,
                padding=0,
            )

    def forward(self, inputs, input_lengths=None):
        if not self.channel_first:
            inputs = inputs.transpose(1, 2).contiguous()
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            outputs = self.conv(inputs)
            if not self.channel_first:
                outputs = outputs.transpose(1, 2).contiguous()
            return outputs, input_lengths * self.stride

        outputs = F.interpolate(inputs, scale_factor=self.stride, mode="nearest")

        if self.use_conv:
            if not self.causal:
                outputs = F.pad(outputs, (self.stride, self.stride))
            else:
                outputs = F.pad(outputs, (self.stride*2, 0))
            outputs = self.conv(outputs)

        if not self.channel_first:
            outputs = outputs.transpose(1, 2).contiguous()
        return outputs, input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, pre_lookahead_len:int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=pre_lookahead_len+1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, inputs, ilens):
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = inputs.transpose(1, 2).contiguous()
        # look ahead
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0)
        outputs = F.leaky_relu(self.conv1(outputs))
        # outputs
        outputs = F.pad(outputs, (2, 0), mode='constant', value=0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        mask = (~make_pad_mask(ilens).unsqueeze(-1).to(inputs.device))
        # residual connection
        outputs = (outputs + inputs) * mask

        return outputs, ilens


class UpsampleConformerEncoder(nn.Module):
    """Progressive upsampling Conformer encoder module.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            upsample_blocks: int = 3,
            upsample_attn_layers: int = 2,
            upsample_ratios: tuple = None,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: Optional[str] = "conv2d",
            normalize_before: bool = True,
            concat_after: bool = False,
            positionwise_layer_type: str = "linear",
            positionwise_conv_kernel_size: int = 3,
            macaron_style: bool = False,
            rel_pos_type: str = "legacy",
            pos_enc_layer_type: str = "rel_pos",
            selfattention_layer_type: str = "rel_selfattn",
            activation_type: str = "swish",
            use_cnn_module: bool = True,
            zero_triu: bool = False,
            cnn_module_kernel: int = 31,
            padding_idx: int = -1,
            causal: bool = False,
            skip: bool = False,
            channel_first: bool = False,
            use_causal_prob: float = None,
            pre_lookahead_len: int = None,
    ):
        super().__init__()
        self._output_size = output_size
        self.causal = causal
        self.skip = skip
        self.channel_first = channel_first
        self.pre_lookahead_len = pre_lookahead_len
        self.use_causal_prob = use_causal_prob

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2dpad":
            self.embed = Conv2dSubsamplingPad(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
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

        if pre_lookahead_len is not None:
            self.pre_lookahead_layer = PreLookaheadLayer(output_size, pre_lookahead_len)

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate=0.0,
            ),
        )
        self.upsample_blocks = nn.ModuleList()
        if upsample_ratios is None:
            upsample_ratios = [2] * upsample_blocks
        self.upsample_ratios = upsample_ratios
        assert upsample_blocks == len(upsample_ratios)
        for i in range(upsample_blocks):
            if not causal:
                upsample_conv_block = Upsample1D(
                    channels=output_size, use_conv=False, use_conv_transpose=True,
                    out_channels=output_size, channel_first=False, stride=upsample_ratios[i], causal=False,
                )
            else:
                upsample_conv_block = Upsample1D(
                    channels=output_size, use_conv=True, use_conv_transpose=False,
                    out_channels=output_size, channel_first=False, stride=upsample_ratios[i], causal=True,
                )
            upsample_attn_block = repeat(
                upsample_attn_layers,
                lambda lnum: EncoderLayer(
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                    convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    stochastic_depth_rate=0.0,
                ),
            )
            attn_input_layer = torch.nn.Sequential(
                torch.nn.Linear(output_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate),
            )
            self.upsample_blocks.append(nn.ModuleList([upsample_conv_block, attn_input_layer, upsample_attn_block]))

        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def rand_mix_masks(self, causal, noncausal):
        use_causal = (torch.rand([causal.shape[0], 1, 1]) <= self.uni_encoder_prob).to(causal)
        masks = use_causal * causal + (1 - use_causal) * noncausal
        return masks

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor = None,
            prev_states: torch.Tensor = None,
            ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        raw_input = xs_pad
        if self.channel_first:
            xs_pad = xs_pad.permute(0, 2, 1)

        if ilens is not None:
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        else:
            masks = torch.ones(
                xs_pad.shape[0], 1, xs_pad.shape[1],
                dtype=torch.bool, device=xs_pad.device
            )

        if self.use_causal_prob is not None:
            use_causal = (torch.rand([xs_pad.shape[0], 1, 1]) <= self.use_causal_prob).to(xs_pad)
        else:
            use_causal = torch.ones([xs_pad.shape[0], 1, 1]).to(xs_pad)

        if self.causal:
            causal_mask = subsequent_mask(
                xs_pad.shape[1], device=xs_pad.device, dtype=masks.dtype
            ).unsqueeze(0)
            causal_mask = masks & causal_mask
            # whether to train causal & non-causal in a single model
            masks = use_causal * causal_mask + (1 - use_causal) * masks

        if (
                isinstance(self.embed, Conv2dSubsampling)
                or isinstance(self.embed, Conv2dSubsampling2)
                or isinstance(self.embed, Conv2dSubsampling6)
                or isinstance(self.embed, Conv2dSubsampling8)
                or isinstance(self.embed, Conv2dSubsamplingPad)
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

        if self.pre_lookahead_len is not None:
            xs = xs_pad
            if isinstance(xs_pad, tuple):
                xs = xs_pad[0]
            xs, _ = self.pre_lookahead_layer(xs, ilens)
            if isinstance(xs_pad, tuple):
                xs_pad = (xs, xs_pad[1])

        # 1. modeling on inputs
        intermediate_outs = []
        xs_pad, masks = self.encoders(xs_pad, masks)

        # 2. progressive upsampling
        outs, olens = xs_pad, ilens
        total_ratio = 1
        for up_ratio, layer in zip(self.upsample_ratios, self.upsample_blocks):
            up_layer, attn_input_layer, attn_layer = layer

            if isinstance(outs, tuple):
                outs = outs[0]
            outs, olens = up_layer(outs, olens)
            masks = (~make_pad_mask(olens)[:, None, :]).to(outs.device)
            total_ratio = total_ratio * up_ratio
            if self.causal:
                causal_mask = causal_block_mask(
                    outs.shape[1], total_ratio, device=outs.device, dtype=masks.dtype
                ).unsqueeze(0)
                causal_mask = masks & causal_mask
                masks = use_causal * causal_mask + (1 - use_causal) * masks
            outs = attn_input_layer(outs)
            outs, _ = attn_layer(outs, masks)

        xs_pad = outs
        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        if self.channel_first:
            xs_pad = xs_pad.permute(0, 2, 1)

        if self.skip:
            xs_pad = xs_pad + raw_input

        # olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None

        if ilens is not None:
            return xs_pad, olens, None
        else:
            return xs_pad

