#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import math
import torch
from typing import Optional, Tuple, Union
from funasr.models.transformer.utils.nets_utils import pad_to_len


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


class RWKVConvInput(torch.nn.Module):
    """Streaming ConvInput module definition.
    Args:
        input_size: Input size.
        conv_size: Convolution size.
        subsampling_factor: Subsampling factor.
        output_size: Block output dimension.
    """

    def __init__(
        self,
        input_size: int,
        conv_size: Union[int, Tuple],
        subsampling_factor: int = 4,
        conv_kernel_size: int = 3,
        output_size: Optional[int] = None,
    ) -> None:
        """Construct a ConvInput object."""
        super().__init__()
        if subsampling_factor == 1:
            conv_size1, conv_size2, conv_size3 = conv_size

            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    1, conv_size1, conv_kernel_size, stride=1, padding=(conv_kernel_size - 1) // 2
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    conv_size1,
                    conv_size1,
                    conv_kernel_size,
                    stride=[1, 2],
                    padding=(conv_kernel_size - 1) // 2,
                ),
                torch.nn.ReLU(),
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
                    stride=[1, 2],
                    padding=(conv_kernel_size - 1) // 2,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    conv_size2,
                    conv_size3,
                    conv_kernel_size,
                    stride=1,
                    padding=(conv_kernel_size - 1) // 2,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    conv_size3,
                    conv_size3,
                    conv_kernel_size,
                    stride=[1, 2],
                    padding=(conv_kernel_size - 1) // 2,
                ),
                torch.nn.ReLU(),
            )

            output_proj = conv_size3 * ((input_size // 2) // 2)

            self.subsampling_factor = 1

            self.stride_1 = 1

            self.create_new_mask = self.create_new_vgg_mask

        else:
            conv_size1, conv_size2, conv_size3 = conv_size

            kernel_1 = int(subsampling_factor / 2)

            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(
                    1, conv_size1, conv_kernel_size, stride=1, padding=(conv_kernel_size - 1) // 2
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    conv_size1,
                    conv_size1,
                    conv_kernel_size,
                    stride=[kernel_1, 2],
                    padding=(conv_kernel_size - 1) // 2,
                ),
                torch.nn.ReLU(),
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
                    stride=[2, 2],
                    padding=(conv_kernel_size - 1) // 2,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    conv_size2,
                    conv_size3,
                    conv_kernel_size,
                    stride=1,
                    padding=(conv_kernel_size - 1) // 2,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    conv_size3,
                    conv_size3,
                    conv_kernel_size,
                    stride=1,
                    padding=(conv_kernel_size - 1) // 2,
                ),
                torch.nn.ReLU(),
            )

            output_proj = conv_size3 * ((input_size // 2) // 2)

            self.subsampling_factor = subsampling_factor

            self.create_new_mask = self.create_new_vgg_mask

            self.stride_1 = kernel_1

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
            return mask[:, ::2][:, :: self.stride_1]
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
