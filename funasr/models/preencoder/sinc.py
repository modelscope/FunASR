#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sinc convolutions for raw audio input."""

from collections import OrderedDict
from funasr.models.preencoder.abs_preencoder import AbsPreEncoder
from funasr.layers.sinc_conv import LogCompression
from funasr.layers.sinc_conv import SincConv
import humanfriendly
import torch
from typing import Optional
from typing import Tuple
from typing import Union


class LightweightSincConvs(AbsPreEncoder):
    """Lightweight Sinc Convolutions.

    Instead of using precomputed features, end-to-end speech recognition
    can also be done directly from raw audio using sinc convolutions, as
    described in "Lightweight End-to-End Speech Recognition from Raw Audio
    Data Using Sinc-Convolutions" by Kürzinger et al.
    https://arxiv.org/abs/2010.07597

    To use Sinc convolutions in your model instead of the default f-bank
    frontend, set this module as your pre-encoder with `preencoder: sinc`
    and use the input of the sliding window frontend with
    `frontend: sliding_window` in your yaml configuration file.
    So that the process flow is:

    Frontend (SlidingWindow) -> SpecAug -> Normalization ->
    Pre-encoder (LightweightSincConvs) -> Encoder -> Decoder

    Note that this method also performs data augmentation in time domain
    (vs. in spectral domain in the default frontend).
    Use `plot_sinc_filters.py` to visualize the learned Sinc filters.
    """

    def __init__(
        self,
        fs: Union[int, str, float] = 16000,
        in_channels: int = 1,
        out_channels: int = 256,
        activation_type: str = "leakyrelu",
        dropout_type: str = "dropout",
        windowing_type: str = "hamming",
        scale_type: str = "mel",
    ):
        """Initialize the module.

        Args:
            fs: Sample rate.
            in_channels: Number of input channels.
            out_channels: Number of output channels (for each input channel).
            activation_type: Choice of activation function.
            dropout_type: Choice of dropout function.
            windowing_type: Choice of windowing function.
            scale_type:  Choice of filter-bank initialization scale.
        """
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        self.fs = fs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_type = activation_type
        self.dropout_type = dropout_type
        self.windowing_type = windowing_type
        self.scale_type = scale_type

        self.choices_dropout = {
            "dropout": torch.nn.Dropout,
            "spatial": SpatialDropout,
            "dropout2d": torch.nn.Dropout2d,
        }
        if dropout_type not in self.choices_dropout:
            raise NotImplementedError(
                f"Dropout type has to be one of "
                f"{list(self.choices_dropout.keys())}",
            )

        self.choices_activation = {
            "leakyrelu": torch.nn.LeakyReLU,
            "relu": torch.nn.ReLU,
        }
        if activation_type not in self.choices_activation:
            raise NotImplementedError(
                f"Activation type has to be one of "
                f"{list(self.choices_activation.keys())}",
            )

        # initialization
        self._create_sinc_convs()
        # Sinc filters require custom initialization
        self.espnet_initialization_fn()

    def _create_sinc_convs(self):
        blocks = OrderedDict()

        # SincConvBlock
        out_channels = 128
        self.filters = SincConv(
            self.in_channels,
            out_channels,
            kernel_size=101,
            stride=1,
            fs=self.fs,
            window_func=self.windowing_type,
            scale_type=self.scale_type,
        )
        block = OrderedDict(
            [
                ("Filters", self.filters),
                ("LogCompression", LogCompression()),
                ("BatchNorm", torch.nn.BatchNorm1d(out_channels, affine=True)),
                ("AvgPool", torch.nn.AvgPool1d(2)),
            ]
        )
        blocks["SincConvBlock"] = torch.nn.Sequential(block)
        in_channels = out_channels

        # First convolutional block, connects the sinc output to the front-end "body"
        out_channels = 128
        blocks["DConvBlock1"] = self.gen_lsc_block(
            in_channels,
            out_channels,
            depthwise_kernel_size=25,
            depthwise_stride=2,
            pointwise_groups=0,
            avgpool=True,
            dropout_probability=0.1,
        )
        in_channels = out_channels

        # Second convolutional block, multiple convolutional layers
        out_channels = self.out_channels
        for layer in [2, 3, 4]:
            blocks[f"DConvBlock{layer}"] = self.gen_lsc_block(
                in_channels, out_channels, depthwise_kernel_size=9, depthwise_stride=1
            )
            in_channels = out_channels

        # Third Convolutional block, acts as coupling to encoder
        out_channels = self.out_channels
        blocks["DConvBlock5"] = self.gen_lsc_block(
            in_channels,
            out_channels,
            depthwise_kernel_size=7,
            depthwise_stride=1,
            pointwise_groups=0,
        )

        self.blocks = torch.nn.Sequential(blocks)

    def gen_lsc_block(
        self,
        in_channels: int,
        out_channels: int,
        depthwise_kernel_size: int = 9,
        depthwise_stride: int = 1,
        depthwise_groups=None,
        pointwise_groups=0,
        dropout_probability: float = 0.15,
        avgpool=False,
    ):
        """Generate a convolutional block for Lightweight Sinc convolutions.

        Each block consists of either a depthwise or a depthwise-separable
        convolutions together with dropout, (batch-)normalization layer, and
        an optional average-pooling layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            depthwise_kernel_size: Kernel size of the depthwise convolution.
            depthwise_stride: Stride of the depthwise convolution.
            depthwise_groups: Number of groups of the depthwise convolution.
            pointwise_groups: Number of groups of the pointwise convolution.
            dropout_probability: Dropout probability in the block.
            avgpool: If True, an AvgPool layer is inserted.

        Returns:
            torch.nn.Sequential: Neural network building block.
        """
        block = OrderedDict()
        if not depthwise_groups:
            # GCD(in_channels, out_channels) to prevent size mismatches
            depthwise_groups, r = in_channels, out_channels
            while r != 0:
                depthwise_groups, r = depthwise_groups, depthwise_groups % r
        block["depthwise"] = torch.nn.Conv1d(
            in_channels,
            out_channels,
            depthwise_kernel_size,
            depthwise_stride,
            groups=depthwise_groups,
        )
        if pointwise_groups:
            block["pointwise"] = torch.nn.Conv1d(
                out_channels, out_channels, 1, 1, groups=pointwise_groups
            )
        block["activation"] = self.choices_activation[self.activation_type]()
        block["batchnorm"] = torch.nn.BatchNorm1d(out_channels, affine=True)
        if avgpool:
            block["avgpool"] = torch.nn.AvgPool1d(2)
        block["dropout"] = self.choices_dropout[self.dropout_type](dropout_probability)
        return torch.nn.Sequential(block)

    def espnet_initialization_fn(self):
        """Initialize sinc filters with filterbank values."""
        self.filters.init_filters()
        for block in self.blocks:
            for layer in block:
                if type(layer) == torch.nn.BatchNorm1d and layer.affine:
                    layer.weight.data[:] = 1.0
                    layer.bias.data[:] = 0.0

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Lightweight Sinc Convolutions.

        The input shall be formatted as (B, T, C_in, D_in)
        with B as batch size, T as time dimension, C_in as channels,
        and D_in as feature dimension.

        The output will then be (B, T, C_out*D_out)
        with C_out and D_out as output dimensions.

        The current module structure only handles D_in=400, so that D_out=1.
        Remark for the multichannel case: C_out is the number of out_channels
        given at initialization multiplied with C_in.
        """
        # Transform input data:
        #   (B, T, C_in, D_in) -> (B*T, C_in, D_in)
        B, T, C_in, D_in = input.size()
        input_frames = input.view(B * T, C_in, D_in)
        output_frames = self.blocks.forward(input_frames)

        # ---TRANSFORM: (B*T, C_out, D_out) -> (B, T, C_out*D_out)
        _, C_out, D_out = output_frames.size()
        output_frames = output_frames.view(B, T, C_out * D_out)
        return output_frames, input_lengths  # no state in this layer

    def output_size(self) -> int:
        """Get the output size."""
        return self.out_channels * self.in_channels


class SpatialDropout(torch.nn.Module):
    """Spatial dropout module.

    Apply dropout to full channels on tensors of input (B, C, D)
    """

    def __init__(
        self,
        dropout_probability: float = 0.15,
        shape: Optional[Union[tuple, list]] = None,
    ):
        """Initialize.

        Args:
            dropout_probability: Dropout probability.
            shape (tuple, list): Shape of input tensors.
        """
        super().__init__()
        if shape is None:
            shape = (0, 2, 1)
        self.dropout = torch.nn.Dropout2d(dropout_probability)
        self.shape = (shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward of spatial dropout module."""
        y = x.permute(*self.shape)
        y = self.dropout(y)
        return y.permute(*self.shape)
