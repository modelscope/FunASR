"""Convolution modules for X-former blocks."""

from typing import Dict, Optional, Tuple

import torch


class ConformerConvolution(torch.nn.Module):
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


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit module definition.

    Args:
        size: Initial size to determine the number of channels.
        kernel_size: Size of the convolving kernel.
        norm_class: Normalization module class.
        norm_args: Normalization module arguments.
        dropout_rate: Dropout rate.
        causal: Whether to use causal convolution (set to True if streaming).

    """

    def __init__(
        self,
        size: int,
        kernel_size: int,
        norm_class: torch.nn.Module = torch.nn.LayerNorm,
        norm_args: Dict = {},
        dropout_rate: float = 0.0,
        causal: bool = False,
    ) -> None:
        """Construct a ConvolutionalSpatialGatingUnit object."""
        super().__init__()

        channels = size // 2

        self.kernel_size = kernel_size

        if causal:
            self.lorder = kernel_size - 1
            padding = 0
        else:
            self.lorder = 0
            padding = (kernel_size - 1) // 2

        self.conv = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
        )

        self.norm = norm_class(channels, **norm_args)
        self.activation = torch.nn.Identity()

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        right_context: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.

        Args:
            x: ConvolutionalSpatialGatingUnit input sequences. (B, T, D_hidden)
            cache: ConvolutionalSpationGatingUnit input cache.
                   (1, conv_kernel, D_hidden)
            right_context: Number of frames in right context.

        Returns:
            x: ConvolutionalSpatialGatingUnit output sequences. (B, T, D_hidden // 2)

        """
        x_r, x_g = x.chunk(2, dim=-1)

        x_g = self.norm(x_g).transpose(1, 2)

        if self.lorder > 0:
            if cache is None:
                x_g = torch.nn.functional.pad(x_g, (self.lorder, 0), "constant", 0.0)
            else:
                x_g = torch.cat([cache, x_g], dim=2)

                if right_context > 0:
                    cache = x_g[:, :, -(self.lorder + right_context) : -right_context]
                else:
                    cache = x_g[:, :, -self.lorder :]

        x_g = self.conv(x_g).transpose(1, 2)

        x = self.dropout(x_r * self.activation(x_g))

        return x, cache
