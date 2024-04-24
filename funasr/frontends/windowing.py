#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sliding Window for raw audio input data."""

import torch
import torch.nn as nn
from typing import Tuple


class SlidingWindow(nn.Module):
    """Sliding Window.
    Provides a sliding window over a batched continuous raw audio tensor.
    Optionally, provides padding (Currently not implemented).
    Combine this module with a pre-encoder compatible with raw audio data,
    for example Sinc convolutions.
    Known issues:
    Output length is calculated incorrectly if audio shorter than win_length.
    WARNING: trailing values are discarded - padding not implemented yet.
    There is currently no additional window function applied to input values.
    """

    def __init__(
        self,
        win_length: int = 400,
        hop_length: int = 160,
        channels: int = 1,
        padding: int = None,
        fs=None,
    ):
        """Initialize.
        Args:
            win_length: Length of frame.
            hop_length: Relative starting point of next frame.
            channels: Number of input channels.
            padding: Padding (placeholder, currently not implemented).
            fs:  Sampling rate (placeholder for compatibility, not used).
        """
        super().__init__()
        self.fs = fs
        self.win_length = win_length
        self.hop_length = hop_length
        self.channels = channels
        self.padding = padding

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a sliding window on the input.
        Args:
            input: Input (B, T, C*D) or (B, T*C*D), with D=C=1.
            input_lengths: Input lengths within batch.
        Returns:
            Tensor: Output with dimensions (B, T, C, D), with D=win_length.
            Tensor: Output lengths within batch.
        """
        input_size = input.size()
        B = input_size[0]
        T = input_size[1]
        C = self.channels
        D = self.win_length
        # (B, T, C) --> (T, B, C)
        continuous = input.view(B, T, C).permute(1, 0, 2)
        windowed = continuous.unfold(0, D, self.hop_length)
        # (T, B, C, D) --> (B, T, C, D)
        output = windowed.permute(1, 0, 2, 3).contiguous()
        # After unfold(), windowed lengths change:
        output_lengths = (input_lengths - self.win_length) // self.hop_length + 1
        return output, output_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the window length."""
        return self.win_length
