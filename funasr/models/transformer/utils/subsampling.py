#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""
import numpy as np
import torch
import torch.nn.functional as F
from funasr.models.transformer.embedding import PositionalEncoding
import logging
from funasr.models.scama.utils import sequence_mask
from funasr.models.transformer.utils.nets_utils import sub_factor_to_params, pad_to_len
from typing import Optional, Tuple, Union
import math


class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling.

    Args:
        message (str): Message for error catch
        actual_size (int): the short size that cannot pass the subsampling
        limit (int): the limit size for subsampling

    """

    def __init__(self, message, actual_size, limit):
        """Construct a TooShortUttError for error handler."""
        super().__init__(message)
        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(ins, size):
    """Check if the utterance is too short for subsampling."""
    if isinstance(ins, Conv2dSubsampling2) and size < 3:
        return True, 3
    if isinstance(ins, Conv2dSubsampling) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling6) and size < 11:
        return True, 11
    if isinstance(ins, Conv2dSubsampling8) and size < 15:
        return True, 15
    return False, -1


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsamplingPad(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsamplingPad, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2, padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2, padding=(0, 0)),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )
        self.pad_fn = torch.nn.ConstantPad1d((0, 4), 0.0)

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.transpose(1, 2)
        x = self.pad_fn(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        x_len = torch.sum(x_mask[:, 0, :], dim=-1)
        x_len = (x_len - 1) // 2 + 1
        x_len = (x_len - 1) // 2 + 1
        mask = sequence_mask(x_len, None, x_len.dtype, x[0].device)
        return x, mask[:, None, :]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2)), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:1]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]


class Conv1dSubsampling(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(
        self,
        idim,
        odim,
        kernel_size,
        stride,
        pad,
        tf2torch_tensor_name_prefix_torch: str = "stride_conv",
        tf2torch_tensor_name_prefix_tf: str = "seq2seq/proj_encoder/downsampling",
    ):
        super(Conv1dSubsampling, self).__init__()
        self.conv = torch.nn.Conv1d(idim, odim, kernel_size, stride)
        self.pad_fn = torch.nn.ConstantPad1d(pad, 0.0)
        self.stride = stride
        self.odim = odim
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf

    def output_size(self) -> int:
        return self.odim

    def forward(self, x, x_len):
        """Subsample x."""
        x = x.transpose(1, 2)  # (b, d ,t)
        x = self.pad_fn(x)
        # x = F.relu(self.conv(x))
        x = F.leaky_relu(self.conv(x), negative_slope=0.0)
        x = x.transpose(1, 2)  # (b, t ,d)

        if x_len is None:

            return x, None
        x_len = (x_len - 1) // self.stride + 1
        return x, x_len


class StreamingConvInput(torch.nn.Module):
    """Streaming ConvInput module definition.
    Args:
        input_size: Input size.
        conv_size: Convolution size.
        subsampling_factor: Subsampling factor.
        vgg_like: Whether to use a VGG-like network.
        output_size: Block output dimension.
    """

    def __init__(
        self,
        input_size: int,
        conv_size: Union[int, Tuple],
        subsampling_factor: int = 4,
        vgg_like: bool = True,
        conv_kernel_size: int = 3,
        output_size: Optional[int] = None,
    ) -> None:
        """Construct a ConvInput object."""
        super().__init__()
        if vgg_like:
            if subsampling_factor == 1:
                conv_size1, conv_size2 = conv_size

                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        1,
                        conv_size1,
                        conv_kernel_size,
                        stride=1,
                        padding=(conv_kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        conv_size1,
                        conv_size1,
                        conv_kernel_size,
                        stride=1,
                        padding=(conv_kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d((1, 2)),
                    torch.nn.Conv2d(
                        conv_size1,
                        conv_size2,
                        conv_kernel_size,
                        stride=1,
                        padding=(conv_kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        conv_size2,
                        conv_size2,
                        conv_kernel_size,
                        stride=1,
                        padding=(conv_kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d((1, 2)),
                )

                output_proj = conv_size2 * ((input_size // 2) // 2)

                self.subsampling_factor = 1

                self.stride_1 = 1

                self.create_new_mask = self.create_new_vgg_mask

            else:
                conv_size1, conv_size2 = conv_size

                kernel_1 = int(subsampling_factor / 2)

                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        1,
                        conv_size1,
                        conv_kernel_size,
                        stride=1,
                        padding=(conv_kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        conv_size1,
                        conv_size1,
                        conv_kernel_size,
                        stride=1,
                        padding=(conv_kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d((kernel_1, 2)),
                    torch.nn.Conv2d(
                        conv_size1,
                        conv_size2,
                        conv_kernel_size,
                        stride=1,
                        padding=(conv_kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        conv_size2,
                        conv_size2,
                        conv_kernel_size,
                        stride=1,
                        padding=(conv_kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d((2, 2)),
                )

                output_proj = conv_size2 * ((input_size // 2) // 2)

                self.subsampling_factor = subsampling_factor

                self.create_new_mask = self.create_new_vgg_mask

                self.stride_1 = kernel_1

        else:
            if subsampling_factor == 1:
                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(1, conv_size, 3, [1, 2], [1, 0]),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(conv_size, conv_size, conv_kernel_size, [1, 2], [1, 0]),
                    torch.nn.ReLU(),
                )

                output_proj = conv_size * (((input_size - 1) // 2 - 1) // 2)

                self.subsampling_factor = subsampling_factor
                self.kernel_2 = conv_kernel_size
                self.stride_2 = 1

                self.create_new_mask = self.create_new_conv2d_mask

            else:
                kernel_2, stride_2, conv_2_output_size = sub_factor_to_params(
                    subsampling_factor,
                    input_size,
                )

                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(1, conv_size, 3, 2, [1, 0]),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        conv_size, conv_size, kernel_2, stride_2, [(kernel_2 - 1) // 2, 0]
                    ),
                    torch.nn.ReLU(),
                )

                output_proj = conv_size * conv_2_output_size

                self.subsampling_factor = subsampling_factor
                self.kernel_2 = kernel_2
                self.stride_2 = stride_2

                self.create_new_mask = self.create_new_conv2d_mask

        self.vgg_like = vgg_like
        self.min_frame_length = 7

        if output_size is not None:
            self.output = torch.nn.Linear(output_proj, output_size)
            self.output_size = output_size
        else:
            self.output = None
            self.output_size = output_proj

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor], chunk_size: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequences.
        Args:
            x: ConvInput input sequences. (B, T, D_feats)
            mask: Mask of input sequences. (B, 1, T)
        Returns:
            x: ConvInput output sequences. (B, sub(T), D_out)
            mask: Mask of output sequences. (B, 1, sub(T))
        """
        if mask is not None:
            mask = self.create_new_mask(mask)
            olens = max(mask.eq(0).sum(1))

        b, t, f = x.size()
        x = x.unsqueeze(1)  # (b. 1. t. f)

        if chunk_size is not None:
            max_input_length = int(
                chunk_size
                * self.subsampling_factor
                * (math.ceil(float(t) / (chunk_size * self.subsampling_factor)))
            )
            x = map(lambda inputs: pad_to_len(inputs, max_input_length, 1), x)
            x = list(x)
            x = torch.stack(x, dim=0)
            N_chunks = max_input_length // (chunk_size * self.subsampling_factor)
            x = x.view(b * N_chunks, 1, chunk_size * self.subsampling_factor, f)

        x = self.conv(x)

        _, c, _, f = x.size()
        if chunk_size is not None:
            x = x.transpose(1, 2).contiguous().view(b, -1, c * f)[:, :olens, :]
        else:
            x = x.transpose(1, 2).contiguous().view(b, -1, c * f)

        if self.output is not None:
            x = self.output(x)

        return x, mask[:, :olens][:, : x.size(1)]

    def create_new_vgg_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create a new mask for VGG output sequences.
        Args:
            mask: Mask of input sequences. (B, T)
        Returns:
            mask: Mask of output sequences. (B, sub(T))
        """
        if self.subsampling_factor > 1:
            vgg1_t_len = mask.size(1) - (mask.size(1) % (self.subsampling_factor // 2))
            mask = mask[:, :vgg1_t_len][:, :: self.subsampling_factor // 2]

            vgg2_t_len = mask.size(1) - (mask.size(1) % 2)
            mask = mask[:, :vgg2_t_len][:, ::2]
        else:
            mask = mask

        return mask

    def create_new_conv2d_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create new conformer mask for Conv2d output sequences.
        Args:
            mask: Mask of input sequences. (B, T)
        Returns:
            mask: Mask of output sequences. (B, sub(T))
        """
        if self.subsampling_factor > 1:
            return mask[:, ::2][:, :: self.stride_2]
        else:
            return mask

    def get_size_before_subsampling(self, size: int) -> int:
        """Return the original size before subsampling for a given size.
        Args:
            size: Number of frames after subsampling.
        Returns:
            : Number of frames before subsampling.
        """
        return size * self.subsampling_factor
