# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

import logging
from typing import Union, Dict, List, Tuple, Optional

import torch
from torch import nn

from funasr.models.ctc.ctc import CTC
from funasr.models.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttentionChunk,
)
from funasr.models.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
    StreamingRelPositionalEncoding,
)
from funasr.models.transformer.layer_norm import LayerNorm
from funasr.models.transformer.utils.multi_layer_conv import Conv1dLinear
from funasr.models.transformer.utils.multi_layer_conv import MultiLayeredConv1d
from funasr.models.transformer.utils.nets_utils import get_activation
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.utils.nets_utils import (
    TooShortUttError,
    check_short_utt,
    make_chunk_mask,
    make_source_mask,
)
from funasr.models.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from funasr.models.transformer.utils.repeat import repeat, MultiBlocks
from funasr.models.transformer.utils.subsampling import Conv2dSubsampling
from funasr.models.transformer.utils.subsampling import Conv2dSubsampling2
from funasr.models.transformer.utils.subsampling import Conv2dSubsampling6
from funasr.models.transformer.utils.subsampling import Conv2dSubsampling8
from funasr.models.transformer.utils.subsampling import TooShortUttError
from funasr.models.transformer.utils.subsampling import check_short_utt
from funasr.models.transformer.utils.subsampling import Conv2dSubsamplingPad
from funasr.models.transformer.utils.subsampling import StreamingConvInput
from funasr.register import tables
import pdb


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(self, x):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

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
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


@tables.register("encoder_classes", "ConformerEncoder")
class ConformerEncoder(nn.Module):
    """Conformer encoder module.

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
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
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
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
    ):
        super().__init__()
        self._output_size = output_size

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
            logging.warning("Using legacy_rel_pos and it will be deprecated in the future.")
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
            self.embed = torch.nn.Sequential(pos_enc_class(output_size, positional_dropout_rate))
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
            logging.warning("Using legacy_rel_selfattn and it will be deprecated in the future.")
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

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks

        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

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
                stochastic_depth_rate[lnum],
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
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
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

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

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            xs_pad, masks = self.encoders(xs_pad, masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x = x + self.conditioning_layer(ctc_out)
                            xs_pad = (x, pos_emb)
                        else:
                            xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None


class CausalConvolution(torch.nn.Module):
    """ConformerConvolution module definition.
    Args:
        channels: The number of channels.
        kernel_size: Size of the convolving kernel.
        activation: Type of activation function.
        norm_args: Normalization module arguments.
        causal: Whether to use causal convolution (set to True if streaming).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        norm_args: Dict = {},
        causal: bool = False,
    ) -> None:
        """Construct an ConformerConvolution object."""
        super().__init__()

        assert (kernel_size - 1) % 2 == 0

        self.kernel_size = kernel_size

        self.pointwise_conv1 = torch.nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if causal:
            self.lorder = kernel_size - 1
            padding = 0
        else:
            self.lorder = 0
            padding = (kernel_size - 1) // 2

        self.depthwise_conv = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
        )
        self.norm = torch.nn.BatchNorm1d(channels, **norm_args)
        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        right_context: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x: ConformerConvolution input sequences. (B, T, D_hidden)
            cache: ConformerConvolution input cache. (1, conv_kernel, D_hidden)
            right_context: Number of frames in right context.
        Returns:
            x: ConformerConvolution output sequences. (B, T, D_hidden)
            cache: ConformerConvolution output cache. (1, conv_kernel, D_hidden)
        """
        x = self.pointwise_conv1(x.transpose(1, 2))
        x = torch.nn.functional.glu(x, dim=1)

        if self.lorder > 0:
            if cache is None:
                x = torch.nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
            else:
                x = torch.cat([cache, x], dim=2)

                if right_context > 0:
                    cache = x[:, :, -(self.lorder + right_context) : -right_context]
                else:
                    cache = x[:, :, -self.lorder :]

        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x).transpose(1, 2)

        return x, cache


class ChunkEncoderLayer(torch.nn.Module):
    """Chunk Conformer module definition.
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
        norm_class: torch.nn.Module = LayerNorm,
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
        x = residual + self.feed_forward_scale * self.dropout(self.feed_forward_macaron(x))

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
        x, conv_cache = self.conv_mod(x, cache=self.cache[1], right_context=right_context)
        x = residual + x
        residual = x

        x = self.norm_feed_forward(x)
        x = residual + self.feed_forward_scale * self.feed_forward(x)

        x = self.norm_final(x)
        self.cache = [att_cache, conv_cache]

        return x, pos_enc


@tables.register("encoder_classes", "ChunkConformerEncoder")
class ConformerChunkEncoder(torch.nn.Module):
    """Encoder module definition.
    Args:
        input_size: Input size.
        body_conf: Encoder body configuration.
        input_conf: Encoder input configuration.
        main_conf: Encoder main configuration.
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
        embed_vgg_like: bool = False,
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
        norm_type: str = "layer_norm",
        cnn_module_kernel: int = 31,
        conv_mod_norm_eps: float = 0.00001,
        conv_mod_norm_momentum: float = 0.1,
        simplified_att_score: bool = False,
        dynamic_chunk_training: bool = False,
        short_chunk_threshold: float = 0.75,
        short_chunk_size: int = 25,
        left_chunk_size: int = 0,
        time_reduction_factor: int = 1,
        unified_model_training: bool = False,
        default_chunk_size: int = 16,
        jitter_range: int = 4,
        subsampling_factor: int = 1,
    ) -> None:
        """Construct an Encoder object."""
        super().__init__()

        self.embed = StreamingConvInput(
            input_size=input_size,
            conv_size=output_size,
            subsampling_factor=subsampling_factor,
            vgg_like=embed_vgg_like,
            output_size=output_size,
        )

        self.pos_enc = StreamingRelPositionalEncoding(
            output_size,
            positional_dropout_rate,
        )

        activation = get_activation(activation_type)

        pos_wise_args = (
            output_size,
            linear_units,
            positional_dropout_rate,
            activation,
        )

        conv_mod_norm_args = {
            "eps": conv_mod_norm_eps,
            "momentum": conv_mod_norm_momentum,
        }

        conv_mod_args = (
            output_size,
            cnn_module_kernel,
            activation,
            conv_mod_norm_args,
            dynamic_chunk_training or unified_model_training,
        )

        mult_att_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            simplified_att_score,
        )

        fn_modules = []
        for _ in range(num_blocks):
            module = lambda: ChunkEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttentionChunk(*mult_att_args),
                PositionwiseFeedForward(*pos_wise_args),
                PositionwiseFeedForward(*pos_wise_args),
                CausalConvolution(*conv_mod_args),
                dropout_rate=dropout_rate,
            )
            fn_modules.append(module)

        self.encoders = MultiBlocks(
            [fn() for fn in fn_modules],
            output_size,
        )

        self._output_size = output_size

        self.dynamic_chunk_training = dynamic_chunk_training
        self.short_chunk_threshold = short_chunk_threshold
        self.short_chunk_size = short_chunk_size
        self.left_chunk_size = left_chunk_size

        self.unified_model_training = unified_model_training
        self.default_chunk_size = default_chunk_size
        self.jitter_range = jitter_range

        self.time_reduction_factor = time_reduction_factor

    def output_size(self) -> int:
        return self._output_size

    def get_encoder_input_raw_size(self, size: int, hop_length: int) -> int:
        """Return the corresponding number of sample for a given chunk size, in frames.
        Where size is the number of features frames after applying subsampling.
        Args:
            size: Number of frames after subsampling.
            hop_length: Frontend's hop length
        Returns:
            : Number of raw samples
        """
        return self.embed.get_size_before_subsampling(size) * hop_length

    def get_encoder_input_size(self, size: int) -> int:
        """Return the corresponding number of sample for a given chunk size, in frames.
        Where size is the number of features frames after applying subsampling.
        Args:
            size: Number of frames after subsampling.
        Returns:
            : Number of raw samples
        """
        return self.embed.get_size_before_subsampling(size)

    def reset_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """Initialize/Reset encoder streaming cache.
        Args:
            left_context: Number of frames in left context.
            device: Device ID.
        """
        return self.encoders.reset_streaming_cache(left_context, device)

    def forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequences.
        Args:
            x: Encoder input features. (B, T_in, F)
            x_len: Encoder input features lengths. (B,)
        Returns:
           x: Encoder outputs. (B, T_out, D_enc)
           x_len: Encoder outputs lenghts. (B,)
        """
        short_status, limit_size = check_short_utt(self.embed.subsampling_factor, x.size(1))

        if short_status:
            raise TooShortUttError(
                f"has {x.size(1)} frames and is too short for subsampling "
                + f"(it needs more than {limit_size} frames), return empty results",
                x.size(1),
                limit_size,
            )

        mask = make_source_mask(x_len).to(x.device)

        if self.unified_model_training:
            if self.training:
                chunk_size = (
                    self.default_chunk_size
                    + torch.randint(-self.jitter_range, self.jitter_range + 1, (1,)).item()
                )
            else:
                chunk_size = self.default_chunk_size
            x, mask = self.embed(x, mask, chunk_size)
            pos_enc = self.pos_enc(x)
            chunk_mask = make_chunk_mask(
                x.size(1),
                chunk_size,
                left_chunk_size=self.left_chunk_size,
                device=x.device,
            )
            x_utt = self.encoders(
                x,
                pos_enc,
                mask,
                chunk_mask=None,
            )
            x_chunk = self.encoders(
                x,
                pos_enc,
                mask,
                chunk_mask=chunk_mask,
            )

            olens = mask.eq(0).sum(1)
            if self.time_reduction_factor > 1:
                x_utt = x_utt[:, :: self.time_reduction_factor, :]
                x_chunk = x_chunk[:, :: self.time_reduction_factor, :]
                olens = torch.floor_divide(olens - 1, self.time_reduction_factor) + 1

            return x_utt, x_chunk, olens

        elif self.dynamic_chunk_training:
            max_len = x.size(1)
            if self.training:
                chunk_size = torch.randint(1, max_len, (1,)).item()

                if chunk_size > (max_len * self.short_chunk_threshold):
                    chunk_size = max_len
                else:
                    chunk_size = (chunk_size % self.short_chunk_size) + 1
            else:
                chunk_size = self.default_chunk_size

            x, mask = self.embed(x, mask, chunk_size)
            pos_enc = self.pos_enc(x)

            chunk_mask = make_chunk_mask(
                x.size(1),
                chunk_size,
                left_chunk_size=self.left_chunk_size,
                device=x.device,
            )
        else:
            x, mask = self.embed(x, mask, None)
            pos_enc = self.pos_enc(x)
            chunk_mask = None
        x = self.encoders(
            x,
            pos_enc,
            mask,
            chunk_mask=chunk_mask,
        )

        olens = mask.eq(0).sum(1)
        if self.time_reduction_factor > 1:
            x = x[:, :: self.time_reduction_factor, :]
            olens = torch.floor_divide(olens - 1, self.time_reduction_factor) + 1

        return x, olens, None

    def full_utt_forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequences.
        Args:
            x: Encoder input features. (B, T_in, F)
            x_len: Encoder input features lengths. (B,)
        Returns:
           x: Encoder outputs. (B, T_out, D_enc)
           x_len: Encoder outputs lenghts. (B,)
        """
        short_status, limit_size = check_short_utt(self.embed.subsampling_factor, x.size(1))

        if short_status:
            raise TooShortUttError(
                f"has {x.size(1)} frames and is too short for subsampling "
                + f"(it needs more than {limit_size} frames), return empty results",
                x.size(1),
                limit_size,
            )

        mask = make_source_mask(x_len).to(x.device)
        x, mask = self.embed(x, mask, None)
        pos_enc = self.pos_enc(x)
        x_utt = self.encoders(
            x,
            pos_enc,
            mask,
            chunk_mask=None,
        )

        if self.time_reduction_factor > 1:
            x_utt = x_utt[:, :: self.time_reduction_factor, :]
        return x_utt

    def simu_chunk_forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
        chunk_size: int = 16,
        left_context: int = 32,
        right_context: int = 0,
    ) -> torch.Tensor:
        short_status, limit_size = check_short_utt(self.embed.subsampling_factor, x.size(1))

        if short_status:
            raise TooShortUttError(
                f"has {x.size(1)} frames and is too short for subsampling "
                + f"(it needs more than {limit_size} frames), return empty results",
                x.size(1),
                limit_size,
            )

        mask = make_source_mask(x_len)

        x, mask = self.embed(x, mask, chunk_size)
        pos_enc = self.pos_enc(x)
        chunk_mask = make_chunk_mask(
            x.size(1),
            chunk_size,
            left_chunk_size=self.left_chunk_size,
            device=x.device,
        )

        x = self.encoders(
            x,
            pos_enc,
            mask,
            chunk_mask=chunk_mask,
        )
        olens = mask.eq(0).sum(1)
        if self.time_reduction_factor > 1:
            x = x[:, :: self.time_reduction_factor, :]

        return x

    def chunk_forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
        processed_frames: torch.tensor,
        chunk_size: int = 16,
        left_context: int = 32,
        right_context: int = 0,
    ) -> torch.Tensor:
        """Encode input sequences as chunks.
        Args:
            x: Encoder input features. (1, T_in, F)
            x_len: Encoder input features lengths. (1,)
            processed_frames: Number of frames already seen.
            left_context: Number of frames in left context.
            right_context: Number of frames in right context.
        Returns:
           x: Encoder outputs. (B, T_out, D_enc)
        """
        mask = make_source_mask(x_len)
        x, mask = self.embed(x, mask, None)

        if left_context > 0:
            processed_mask = (
                torch.arange(left_context, device=x.device).view(1, left_context).flip(1)
            )
            processed_mask = processed_mask >= processed_frames
            mask = torch.cat([processed_mask, mask], dim=1)
        pos_enc = self.pos_enc(x, left_context=left_context)
        x = self.encoders.chunk_forward(
            x,
            pos_enc,
            mask,
            chunk_size=chunk_size,
            left_context=left_context,
            right_context=right_context,
        )

        if right_context > 0:
            x = x[:, 0:-right_context, :]

        if self.time_reduction_factor > 1:
            x = x[:, :: self.time_reduction_factor, :]
        return x
