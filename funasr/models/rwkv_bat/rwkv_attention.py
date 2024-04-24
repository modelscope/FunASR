#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import math
import torch
from pathlib import Path
from importlib.util import find_spec
from typing import List, Optional, Tuple, Union


wkv_kernel_encoder = None
wkv_kernel_decoder = None


class WKVLinearAttentionEncoder(torch.autograd.Function):
    """WKVLinearAttention function definition."""

    @staticmethod
    def forward(
        ctx,
        time_decay: torch.Tensor,
        time_first: torch.Tensor,
        key: torch.Tensor,
        value: torch.tensor,
    ) -> torch.Tensor:
        """WKVLinearAttention function forward pass.

        Args:
            time_decay: Channel-wise time decay vector. (D_att)
            time_first: Channel-wise time first vector. (D_att)
            key: Key tensor. (B, U, D_att)
            value: Value tensor. (B, U, D_att)

        Returns:
            out: Weighted Key-Value tensor. (B, U, D_att)

        """
        batch, length, dim = key.size()

        assert length <= wkv_kernel_encoder.context_size, (
            f"Cannot process key of length {length} while context_size "
            f"is ({wkv_kernel_encoder.context_size}). Limit should be increased."
        )

        assert batch * dim % min(dim, 32) == 0, (
            f"batch size ({batch}) by dimension ({dim}) should be a multiple of " f"{min(dim, 32)}"
        )

        ctx.input_dtype = key.dtype

        time_decay = -torch.exp(time_decay.float().contiguous())
        time_first = time_first.float().contiguous()

        key = key.float().contiguous()
        value = value.float().contiguous()

        out = torch.empty_like(key, memory_format=torch.contiguous_format)

        wkv_kernel_encoder.forward(time_decay, time_first, key, value, out)
        ctx.save_for_backward(time_decay, time_first, key, value, out)

        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """WKVLinearAttention function backward pass.

        Args:
            grad_output: Output gradient. (B, U, D_att)

        Returns:
            grad_time_decay: Gradient for channel-wise time decay vector. (D_att)
            grad_time_first: Gradient for channel-wise time first vector. (D_att)
            grad_key: Gradient for key tensor. (B, U, D_att)
            grad_value: Gradient for value tensor. (B, U, D_att)

        """
        time_decay, time_first, key, value, output = ctx.saved_tensors
        grad_dtype = ctx.input_dtype

        batch, _, dim = key.size()

        grad_time_decay = torch.empty(
            (batch, dim),
            memory_format=torch.contiguous_format,
            dtype=time_decay.dtype,
            device=time_decay.device,
        )

        grad_time_first = torch.empty(
            (batch, dim),
            memory_format=torch.contiguous_format,
            dtype=time_decay.dtype,
            device=time_decay.device,
        )

        grad_key = torch.empty_like(key, memory_format=torch.contiguous_format)
        grad_value = torch.empty_like(value, memory_format=torch.contiguous_format)

        wkv_kernel_encoder.backward(
            time_decay,
            time_first,
            key,
            value,
            output,
            grad_output.contiguous(),
            grad_time_decay,
            grad_time_first,
            grad_key,
            grad_value,
        )

        grad_time_decay = torch.sum(grad_time_decay, dim=0)
        grad_time_first = torch.sum(grad_time_first, dim=0)

        return (
            grad_time_decay,
            grad_time_first,
            grad_key,
            grad_value,
        )


class WKVLinearAttentionDecoder(torch.autograd.Function):
    """WKVLinearAttention function definition."""

    @staticmethod
    def forward(
        ctx,
        time_decay: torch.Tensor,
        time_first: torch.Tensor,
        key: torch.Tensor,
        value: torch.tensor,
    ) -> torch.Tensor:
        """WKVLinearAttention function forward pass.

        Args:
            time_decay: Channel-wise time decay vector. (D_att)
            time_first: Channel-wise time first vector. (D_att)
            key: Key tensor. (B, U, D_att)
            value: Value tensor. (B, U, D_att)

        Returns:
            out: Weighted Key-Value tensor. (B, U, D_att)

        """
        batch, length, dim = key.size()

        assert length <= wkv_kernel_decoder.context_size, (
            f"Cannot process key of length {length} while context_size "
            f"is ({wkv_kernel.context_size}). Limit should be increased."
        )

        assert batch * dim % min(dim, 32) == 0, (
            f"batch size ({batch}) by dimension ({dim}) should be a multiple of " f"{min(dim, 32)}"
        )

        ctx.input_dtype = key.dtype

        time_decay = -torch.exp(time_decay.float().contiguous())
        time_first = time_first.float().contiguous()

        key = key.float().contiguous()
        value = value.float().contiguous()

        out = torch.empty_like(key, memory_format=torch.contiguous_format)

        wkv_kernel_decoder.forward(time_decay, time_first, key, value, out)
        ctx.save_for_backward(time_decay, time_first, key, value, out)

        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """WKVLinearAttention function backward pass.

        Args:
            grad_output: Output gradient. (B, U, D_att)

        Returns:
            grad_time_decay: Gradient for channel-wise time decay vector. (D_att)
            grad_time_first: Gradient for channel-wise time first vector. (D_att)
            grad_key: Gradient for key tensor. (B, U, D_att)
            grad_value: Gradient for value tensor. (B, U, D_att)

        """
        time_decay, time_first, key, value, output = ctx.saved_tensors
        grad_dtype = ctx.input_dtype

        batch, _, dim = key.size()

        grad_time_decay = torch.empty(
            (batch, dim),
            memory_format=torch.contiguous_format,
            dtype=time_decay.dtype,
            device=time_decay.device,
        )

        grad_time_first = torch.empty(
            (batch, dim),
            memory_format=torch.contiguous_format,
            dtype=time_decay.dtype,
            device=time_decay.device,
        )

        grad_key = torch.empty_like(key, memory_format=torch.contiguous_format)
        grad_value = torch.empty_like(value, memory_format=torch.contiguous_format)

        wkv_kernel_decoder.backward(
            time_decay,
            time_first,
            key,
            value,
            output,
            grad_output.contiguous(),
            grad_time_decay,
            grad_time_first,
            grad_key,
            grad_value,
        )

        grad_time_decay = torch.sum(grad_time_decay, dim=0)
        grad_time_first = torch.sum(grad_time_first, dim=0)

        return (
            grad_time_decay,
            grad_time_first,
            grad_key,
            grad_value,
        )


def load_encoder_wkv_kernel(context_size: int) -> None:
    """Load WKV CUDA kernel.

    Args:
        context_size: Context size.

    """
    from torch.utils.cpp_extension import load

    global wkv_kernel_encoder

    if wkv_kernel_encoder is not None and wkv_kernel_encoder.context_size == context_size:
        return

    if find_spec("ninja") is None:
        raise ImportError(
            "Ninja package was not found. WKV kernel module can't be loaded "
            "for training. Please, 'pip install ninja' in your environment."
        )

    if not torch.cuda.is_available():
        raise ImportError(
            "CUDA is currently a requirement for WKV kernel loading. "
            "Please set your devices properly and launch again."
        )

    kernel_folder = Path(__file__).resolve().parent / "cuda_encoder"
    kernel_files = [kernel_folder / f for f in ["wkv_op.cpp", "wkv_cuda.cu"]]

    kernel_cflags = [
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        f"-DTmax={context_size}",
    ]
    wkv_kernel_encoder = load(
        name=f"encoder_wkv_{context_size}",
        sources=kernel_files,
        verbose=True,
        extra_cuda_cflags=kernel_cflags,
    )
    wkv_kernel_encoder.context_size = context_size


def load_decoder_wkv_kernel(context_size: int) -> None:
    """Load WKV CUDA kernel.

    Args:
        context_size: Context size.

    """
    from torch.utils.cpp_extension import load

    global wkv_kernel_decoder

    if wkv_kernel_decoder is not None and wkv_kernel_decoder.context_size == context_size:
        return

    if find_spec("ninja") is None:
        raise ImportError(
            "Ninja package was not found. WKV kernel module can't be loaded "
            "for training. Please, 'pip install ninja' in your environment."
        )

    if not torch.cuda.is_available():
        raise ImportError(
            "CUDA is currently a requirement for WKV kernel loading. "
            "Please set your devices properly and launch again."
        )

    kernel_folder = Path(__file__).resolve().parent / "cuda_decoder"
    kernel_files = [kernel_folder / f for f in ["wkv_op.cpp", "wkv_cuda.cu"]]

    kernel_cflags = [
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        f"-DTmax={context_size}",
    ]
    wkv_kernel_decoder = load(
        name=f"decoder_wkv_{context_size}",
        sources=kernel_files,
        verbose=True,
        extra_cuda_cflags=kernel_cflags,
    )
    wkv_kernel_decoder.context_size = context_size


class SelfAttention(torch.nn.Module):
    """SelfAttention module definition.

    Args:
        size: Input/Output size.
        attention_size: Attention hidden size.
        context_size: Context size for WKV kernel.
        block_id: Block index.
        num_blocks: Number of blocks in the architecture.

    """

    def __init__(
        self,
        size: int,
        attention_size: int,
        block_id: int,
        dropout_rate: float,
        num_blocks: int,
    ) -> None:
        """Construct a SelfAttention object."""
        super().__init__()
        self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))

        self.time_decay = torch.nn.Parameter(torch.empty(attention_size))
        self.time_first = torch.nn.Parameter(torch.empty(attention_size))

        self.time_mix_key = torch.nn.Parameter(torch.empty(1, 1, size))
        self.time_mix_value = torch.nn.Parameter(torch.empty(1, 1, size))
        self.time_mix_receptance = torch.nn.Parameter(torch.empty(1, 1, size))

        self.proj_key = torch.nn.Linear(size, attention_size, bias=True)
        self.proj_value = torch.nn.Linear(size, attention_size, bias=True)
        self.proj_receptance = torch.nn.Linear(size, attention_size, bias=True)

        self.proj_output = torch.nn.Linear(attention_size, size, bias=True)

        self.block_id = block_id

        self.reset_parameters(size, attention_size, block_id, num_blocks)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def reset_parameters(
        self, size: int, attention_size: int, block_id: int, num_blocks: int
    ) -> None:
        """Reset module parameters.

        Args:
            size: Block size.
            attention_size: Attention hidden size.
            block_id: Block index.
            num_blocks: Number of blocks in the architecture.

        """
        ratio_0_to_1 = block_id / (num_blocks - 1)
        ratio_1_to_almost0 = 1.0 - (block_id / num_blocks)

        time_weight = torch.ones(1, 1, size)

        for i in range(size):
            time_weight[0, 0, i] = i / size

        decay_speed = [
            -5 + 8 * (h / (attention_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            for h in range(attention_size)
        ]
        decay_speed = torch.tensor(
            decay_speed, dtype=self.time_decay.dtype, device=self.time_decay.device
        )

        zigzag = (
            torch.tensor(
                [(i + 1) % 3 - 1 for i in range(attention_size)],
                dtype=self.time_first.dtype,
                device=self.time_first.device,
            )
            * 0.5
        )

        with torch.no_grad():
            self.time_decay.data = decay_speed
            self.time_first.data = torch.ones_like(self.time_first * math.log(0.3) + zigzag)

            self.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
            self.time_mix_value.data = (
                torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
            )
            self.time_mix_receptance.data = torch.pow(time_weight, 0.5 * ratio_1_to_almost0)

    @torch.no_grad()
    def wkv_linear_attention(
        self,
        time_decay: torch.Tensor,
        time_first: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute WKV with state (i.e.: for inference).

        Args:
            time_decay: Channel-wise time decay vector. (D_att)
            time_first: Channel-wise time first vector. (D_att)
            key: Key tensor. (B, 1, D_att)
            value: Value tensor. (B, 1, D_att)
            state: Decoder hidden states. [3 x (B, D_att)]

        Returns:
            output: Weighted Key-Value. (B, 1, D_att)
            state: Decoder hidden states. [3 x (B, 1, D_att)]

        """
        num_state, den_state, max_state = state
        time_decay = -torch.exp(time_decay)
        max_for_output = torch.maximum(max_state, (time_first + key))

        e1 = torch.exp(max_state - max_for_output)
        e2 = torch.exp((time_first + key) - max_for_output)

        numerator = e1 * num_state + e2 * value
        denominator = e1 * den_state + e2

        max_for_state = torch.maximum(key, (max_state + time_decay))

        e1 = torch.exp((max_state + time_decay) - max_for_state)
        e2 = torch.exp(key - max_for_state)

        wkv = numerator / denominator

        state = [e1 * num_state + e2 * value, e1 * den_state + e2, max_for_state]

        return wkv, state


class DecoderSelfAttention(SelfAttention):
    """SelfAttention module definition.

    Args:
        size: Input/Output size.
        attention_size: Attention hidden size.
        context_size: Context size for WKV kernel.
        block_id: Block index.
        num_blocks: Number of blocks in the architecture.

    """

    def __init__(
        self,
        size: int,
        attention_size: int,
        context_size: int,
        block_id: int,
        dropout_rate: float,
        num_blocks: int,
    ) -> None:
        """Construct a SelfAttention object."""
        super().__init__(size, attention_size, block_id, dropout_rate, num_blocks)
        # load_decoder_wkv_kernel(context_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Compute time mixing.

        Args:
            x: SelfAttention input sequences. (B, U, size)
            state: Decoder hidden states. [5 x (B, 1, D_att, N)]

        Returns:
            x: SelfAttention output sequences. (B, U, size)

        """
        shifted_x = self.time_shift(x) if state is None else state[1][..., self.block_id]

        key = x * self.time_mix_key + shifted_x * (1 - self.time_mix_key)
        value = x * self.time_mix_value + shifted_x * (1 - self.time_mix_value)
        receptance = x * self.time_mix_receptance + shifted_x * (1 - self.time_mix_receptance)

        key = self.proj_key(key)
        value = self.proj_value(value)
        receptance = torch.sigmoid(self.proj_receptance(receptance))

        if state is not None:
            state[1][..., self.block_id] = x

            wkv, att_state = self.wkv_linear_attention(
                self.time_decay,
                self.time_first,
                key,
                value,
                tuple(s[..., self.block_id] for s in state[2:]),
            )

            state[2][..., self.block_id] = att_state[0]
            state[3][..., self.block_id] = att_state[1]
            state[4][..., self.block_id] = att_state[2]
        else:
            wkv = WKVLinearAttentionDecoder.apply(self.time_decay, self.time_first, key, value)

        wkv = self.dropout(wkv)
        x = self.proj_output(receptance * wkv)

        return x, state


class EncoderSelfAttention(SelfAttention):
    """SelfAttention module definition.

    Args:
        size: Input/Output size.
        attention_size: Attention hidden size.
        context_size: Context size for WKV kernel.
        block_id: Block index.
        num_blocks: Number of blocks in the architecture.

    """

    def __init__(
        self,
        size: int,
        attention_size: int,
        context_size: int,
        block_id: int,
        dropout_rate: float,
        num_blocks: int,
    ) -> None:
        """Construct a SelfAttention object."""
        super().__init__(size, attention_size, block_id, dropout_rate, num_blocks)
        # load_encoder_wkv_kernel(context_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Compute time mixing.

        Args:
            x: SelfAttention input sequences. (B, U, size)
            state: Decoder hidden states. [5 x (B, 1, D_att, N)]

        Returns:
            x: SelfAttention output sequences. (B, U, size)

        """
        shifted_x = self.time_shift(x) if state is None else state[1][..., self.block_id]

        key = x * self.time_mix_key + shifted_x * (1 - self.time_mix_key)
        value = x * self.time_mix_value + shifted_x * (1 - self.time_mix_value)
        receptance = x * self.time_mix_receptance + shifted_x * (1 - self.time_mix_receptance)

        key = self.proj_key(key)
        value = self.proj_value(value)
        receptance = torch.sigmoid(self.proj_receptance(receptance))

        if state is not None:
            state[1][..., self.block_id] = x

            wkv, att_state = self.wkv_linear_attention(
                self.time_decay,
                self.time_first,
                key,
                value,
                tuple(s[..., self.block_id] for s in state[2:]),
            )

            state[2][..., self.block_id] = att_state[0]
            state[3][..., self.block_id] = att_state[1]
            state[4][..., self.block_id] = att_state[2]
        else:
            wkv = WKVLinearAttentionEncoder.apply(self.time_decay, self.time_first, key, value)

        wkv = self.dropout(wkv)
        x = self.proj_output(receptance * wkv)

        return x, state
