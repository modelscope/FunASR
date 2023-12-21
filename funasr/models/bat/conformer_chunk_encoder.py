
"""Conformer encoder definition."""

import logging
from typing import Union, Dict, List, Tuple, Optional

import torch
from torch import nn


from funasr.models.bat.attention import (
    RelPositionMultiHeadedAttentionChunk,
)
from funasr.models.transformer.embedding import (
    StreamingRelPositionalEncoding,
)
from funasr.models.transformer.layer_norm import LayerNorm
from funasr.models.transformer.utils.nets_utils import get_activation
from funasr.models.transformer.utils.nets_utils import (
    TooShortUttError,
    check_short_utt,
    make_chunk_mask,
    make_source_mask,
)
from funasr.models.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from funasr.models.transformer.utils.repeat import repeat, MultiBlocks
from funasr.models.transformer.utils.subsampling import TooShortUttError
from funasr.models.transformer.utils.subsampling import check_short_utt
from funasr.models.transformer.utils.subsampling import StreamingConvInput
from funasr.register import tables



class ChunkEncoderLayer(nn.Module):
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



class CausalConvolution(nn.Module):
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

@tables.register("encoder_classes", "ConformerChunkEncoder")
class ConformerChunkEncoder(nn.Module):
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
            input_size,
            output_size,
            subsampling_factor,
            vgg_like=embed_vgg_like,
            output_size=output_size,
        )

        self.pos_enc = StreamingRelPositionalEncoding(
            output_size,
            positional_dropout_rate,
        )

        activation = get_activation(
            activation_type
       )

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
        short_status, limit_size = check_short_utt(
            self.embed.subsampling_factor, x.size(1)
        )

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
                chunk_size = self.default_chunk_size + torch.randint(-self.jitter_range, self.jitter_range+1, (1,)).item()
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
                x_utt = x_utt[:,::self.time_reduction_factor,:]
                x_chunk = x_chunk[:,::self.time_reduction_factor,:]
                olens = torch.floor_divide(olens-1, self.time_reduction_factor) + 1

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
            x = x[:,::self.time_reduction_factor,:]
            olens = torch.floor_divide(olens-1, self.time_reduction_factor) + 1

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
        short_status, limit_size = check_short_utt(
            self.embed.subsampling_factor, x.size(1)
        )

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
            x_utt = x_utt[:,::self.time_reduction_factor,:]
        return x_utt

    def simu_chunk_forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
        chunk_size: int = 16,
        left_context: int = 32,
        right_context: int = 0,
    ) -> torch.Tensor:
        short_status, limit_size = check_short_utt(
            self.embed.subsampling_factor, x.size(1)
        )

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
            x = x[:,::self.time_reduction_factor,:]

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
                torch.arange(left_context, device=x.device)
                .view(1, left_context)
                .flip(1)
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
            x = x[:,::self.time_reduction_factor,:]
        return x
