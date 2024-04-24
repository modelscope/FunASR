#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
from typing import List, Optional, Tuple


class FeedForward(torch.nn.Module):
    """FeedForward module definition.

    Args:
        size: Input/Output size.
        hidden_size: Hidden size.
        block_id: Block index.
        num_blocks: Number of blocks in the architecture.

    """

    def __init__(
        self, size: int, hidden_size: int, block_id: int, dropout_rate: float, num_blocks: int
    ) -> None:
        """Construct a FeedForward object."""
        super().__init__()

        self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))

        self.time_mix_key = torch.nn.Parameter(torch.empty(1, 1, size))
        self.time_mix_receptance = torch.nn.Parameter(torch.empty(1, 1, size))

        self.proj_key = torch.nn.Linear(size, hidden_size, bias=True)
        self.proj_value = torch.nn.Linear(hidden_size, size, bias=True)
        self.proj_receptance = torch.nn.Linear(size, size, bias=True)

        self.block_id = block_id

        self.reset_parameters(size, block_id, num_blocks)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def reset_parameters(self, size: int, block_id: int, num_blocks: int) -> None:
        """Reset module parameters.

        Args:
            size: Block size.
            block_id: Block index.
            num_blocks: Number of blocks in the architecture.

        """
        ratio_1_to_almost0 = 1.0 - (block_id / num_blocks)

        time_weight = torch.ones(1, 1, size)

        for i in range(size):
            time_weight[0, 0, i] = i / size

        with torch.no_grad():
            self.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
            self.time_mix_receptance.data = torch.pow(time_weight, ratio_1_to_almost0)

    def forward(
        self, x: torch.Tensor, state: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Compute channel mixing.

        Args:
            x: FeedForward input sequences. (B, U, size)
            state: Decoder hidden state. [5 x (B, 1, size, N)]

        Returns:
            x: FeedForward output sequences. (B, U, size)
            state: Decoder hidden state. [5 x (B, 1, size, N)]

        """
        shifted_x = self.time_shift(x) if state is None else state[0][..., self.block_id]

        key = x * self.time_mix_key + shifted_x * (1 - self.time_mix_key)
        receptance = x * self.time_mix_receptance + shifted_x * (1 - self.time_mix_receptance)

        key = torch.square(torch.relu(self.proj_key(key)))
        value = self.proj_value(self.dropout(key))
        receptance = torch.sigmoid(self.proj_receptance(receptance))

        if state is not None:
            state[0][..., self.block_id] = x

        x = receptance * value

        return x, state
