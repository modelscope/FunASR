# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2018-2019, Mingkun Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import math

import torch
from numba import cuda

from funasr.modules.rnnt_multi_blank.utils import global_constants, rnnt_helper

warp_size = global_constants.warp_size()
dtype = global_constants.dtype()

CTA_REDUCE_SIZE = 128


class I_Op(enum.Enum):
    """
    Represents an operation that is performed on the input tensor
    """

    EXPONENTIAL = 0
    IDENTITY = 1


class R_Op(enum.Enum):
    """
    Represents a reduction operation performed on the input tensor
    """

    ADD = 0
    MAXIMUM = 1


@cuda.jit(device=True)
def CTAReduce(tid: int, x, storage, count: int, R_opid: int):
    """
    CUDA Warp reduction kernel.

    It is a device kernel to be called by other kernels.

    The data will be read from the right segement recursively, and reduced (ROP) onto
    the left half. Operation continues while warp size is larger than a given offset.
    Beyond this offset, warp reduction is performed via `shfl_down_sync`,
    which halves the reduction space and sums the two halves at each call.

    Note:
        Efficient warp occurs at input shapes of 2 ^ K.

    References:
        - Warp Primitives
          [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/]

    Args:
        tid: CUDA thread index
        x: activation. Single float.
        storage: shared memory of size CTA_REDUCE_SIZE used for reduction
            in parallel threads.
        count: equivalent to num_rows, which is equivalent to alphabet_size (V+1)
        R_opid: Operator ID for reduction. See R_Op for more information.
    """

    storage[tid] = x

    cuda.syncthreads()

    # Fold the data in half with each pass
    offset = CTA_REDUCE_SIZE // 2
    while offset >= warp_size:
        if (tid + offset) < count and tid < offset:
            # Read from the right half and store to the left half.
            if R_opid == 0:
                x = rnnt_helper.add(x, storage[offset + tid])
            else:
                x = rnnt_helper.maximum(x, storage[offset + tid])

            storage[tid] = x

        cuda.syncthreads()
        offset = offset // 2

    offset = warp_size // 2
    while offset > 0:
        # warp reduction and sync
        shuff = cuda.shfl_down_sync(0xFFFFFFFF, x, offset)

        if (tid + offset < count) and (tid < offset):
            if R_opid == 0:
                x = rnnt_helper.add(x, shuff)
            else:
                x = rnnt_helper.maximum(x, shuff)

        offset = offset // 2

    return x


@cuda.jit()
def _reduce_rows(I_opid: int, R_opid: int, acts, output, num_rows: int):
    """
    CUDA Warp reduction kernel which reduces via the R_Op.Maximum

    Reduces the input data such that I_Op = Identity and R_op = Maximum.
    The result is stored in the blockIdx, and is stored as an identity op.

    Note:
        Efficient warp occurs at input shapes of 2 ^ K.

    References:
        - Warp Primitives
          [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/]

    Args:
        I_opid: Operator ID for input. See I_Op for more information. For this kernel,
            the Identity op is chosen in general, and therefore the input
            is reduced in place without scaling.
        R_opid: Operator ID for reduction. See R_Op for more information.
            For this kernel, generally Maximum op is chosen.
            It reduces the kernel via max.
        acts: Flatened activation matrix of shape [B * T * U * (V+1)].
        output: Flatened output matrix of shape [B * T * U * (V+1)].
            Data will be overwritten.
        num_rows: Vocabulary size (including blank token) - V+1.
    """

    tid = cuda.threadIdx.x
    idx = tid
    col = cuda.blockIdx.x

    # allocate shared thread memory
    storage = cuda.shared.array(shape=(CTA_REDUCE_SIZE,), dtype=acts.dtype)

    max = output[col]

    # // Each block works on a column
    if idx < num_rows:
        curr = acts[col * num_rows + idx] - max
        if I_opid == 0:
            curr = rnnt_helper.exponential(curr)
        else:
            curr = rnnt_helper.identity(curr)

    idx += CTA_REDUCE_SIZE

    while idx < num_rows:
        activation_ = acts[col * num_rows + idx] - max

        if I_opid == 0 and R_opid == 0:
            curr = rnnt_helper.add(curr, rnnt_helper.exponential(activation_))
        elif I_opid == 0 and R_opid == 1:
            curr = rnnt_helper.maximum(curr, rnnt_helper.exponential(activation_))
        elif I_opid == 1 and R_opid == 0:
            curr = rnnt_helper.add(curr, rnnt_helper.identity(activation_))
        else:
            curr = rnnt_helper.maximum(curr, rnnt_helper.identity(activation_))

        idx += CTA_REDUCE_SIZE

    # // Sum thread-totals over the CTA.
    curr = CTAReduce(tid, curr, storage, num_rows, R_opid)

    # // Store result in out (inplace, I_op: identity)
    if tid == 0:
        output[col] = curr


@cuda.jit()
def _reduce_minus(I_opid: int, R_opid: int, acts, output, num_rows: int):
    """
    CUDA Warp reduction kernel which reduces via the R_Op.Add

    Reduces the input data such that I_Op = Exponential and R_op = Add.
    The result is stored in the blockIdx, and is stored as an exp op.

    Note:
        Efficient warp occurs at input shapes of 2 ^ K.

    References:
        - Warp Primitives
          [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/]

    Args:
        I_opid: Operator ID for input. See I_Op for more information. For this kernel,
            the Exponential op is chosen in general, and therefore the input
            is reduced in place with scaling.
        R_opid: Operator ID for reduction. See R_Op for more information. For this
            kernel, generally Add op is chosen. It reduces the kernel via summation.
        acts: Flatened activation matrix of shape [B * T * U * (V+1)].
        output: Flatened output matrix of shape [B * T * U * (V+1)].
            Data will be overwritten.
        num_rows: Vocabulary size (including blank token) - V+1.
    """

    tid = cuda.threadIdx.x
    idx = tid
    col = cuda.blockIdx.x

    # allocate shared thread memory
    storage = cuda.shared.array(shape=(CTA_REDUCE_SIZE,), dtype=acts.dtype)

    max = output[col]

    # // Each block works on a column
    if idx < num_rows:
        curr = acts[col * num_rows + idx] - max
        if I_opid == 0:
            curr = rnnt_helper.exponential(curr)
        else:
            curr = rnnt_helper.identity(curr)

    idx += CTA_REDUCE_SIZE

    while idx < num_rows:
        activation_ = acts[col * num_rows + idx] - max

        if I_opid == 0 and R_opid == 0:
            curr = rnnt_helper.add(curr, rnnt_helper.exponential(activation_))
        elif I_opid == 0 and R_opid == 1:
            curr = rnnt_helper.maximum(curr, rnnt_helper.exponential(activation_))
        elif I_opid == 1 and R_opid == 0:
            curr = rnnt_helper.add(curr, rnnt_helper.identity(activation_))
        else:
            curr = rnnt_helper.maximum(curr, rnnt_helper.identity(activation_))

        idx += CTA_REDUCE_SIZE

    # // Sum thread-totals over the CTA.
    curr = CTAReduce(tid, curr, storage, num_rows, R_opid)

    # // Store result in out (inplace, I_op: exponential)
    if tid == 0:
        output[col] = -max - math.log(curr)


def ReduceHelper(
    I_opid: int,
    R_opid: int,
    acts: torch.Tensor,
    output: torch.Tensor,
    num_rows: int,
    num_cols: int,
    minus: bool,
    stream,
):
    """
    CUDA Warp reduction kernel helper which reduces via the R_Op.Add and writes
    the result to `output` according to I_op id.

    The result is stored in the blockIdx.

    Note:
        Efficient warp occurs at input shapes of 2 ^ K.

    References:
        - Warp Primitives
          [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/]

    Args:
        I_opid: Operator ID for input. See I_Op for more information.
        R_opid: Operator ID for reduction. See R_Op for more information.
        acts: Flatened activation matrix of shape [B * T * U * (V+1)].
        output: Flatened output matrix of shape [B * T * U * (V+1)].
            Data will be overwritten.
        num_rows: Vocabulary size (including blank token) - V+1.
            Represents the number of threads per block.
        num_cols: Flattened shape of activation matrix, without vocabulary dimension
            (B * T * U). Represents number of blocks per grid.
        minus: Bool flag whether to add or subtract as reduction.
            If minus is set; calls _reduce_minus, else calls _reduce_rows kernel.
        stream: CUDA Stream.
    """

    if minus:
        grid_size = num_cols
        # call kernel
        _reduce_minus[grid_size, CTA_REDUCE_SIZE, stream, 0](
            I_opid, R_opid, acts, output, num_rows
        )

    else:
        grid_size = num_cols
        # call kernel
        _reduce_rows[grid_size, CTA_REDUCE_SIZE, stream, 0](
            I_opid, R_opid, acts, output, num_rows
        )

    return True


def reduce_exp(acts: torch.Tensor, denom, rows: int, cols: int, minus: bool, stream):
    """
    Helper method to call the Warp Reduction Kernel to perform `exp` reduction.

    Note:
        Efficient warp occurs at input shapes of 2 ^ K.

    References:
        - Warp Primitives
          [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/]

    Args:
        acts: Flatened activation matrix of shape [B * T * U * (V+1)].
        output: Flatened output matrix of shape [B * T * U * (V+1)].
            Data will be overwritten.
        rows: Vocabulary size (including blank token) - V+1.
            Represents the number of threads per block.
        cols: Flattened shape of activation matrix, without vocabulary dimension
            (B * T * U). Represents number of blocks per grid.
        minus: Bool flag whether to add or subtract as reduction.
            If minus is set; calls _reduce_minus, else calls _reduce_rows kernel.
        stream: CUDA Stream.
    """

    return ReduceHelper(
        I_opid=I_Op.EXPONENTIAL.value,
        R_opid=R_Op.ADD.value,
        acts=acts,
        output=denom,
        num_rows=rows,
        num_cols=cols,
        minus=minus,
        stream=stream,
    )


def reduce_max(acts: torch.Tensor, denom, rows: int, cols: int, minus: bool, stream):
    """
    Helper method to call the Warp Reduction Kernel to perform `max` reduction.

    Note:
        Efficient warp occurs at input shapes of 2 ^ K.

    References:
        - Warp Primitives
          [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/]

    Args:
        acts: Flatened activation matrix of shape [B * T * U * (V+1)].
        output: Flatened output matrix of shape [B * T * U * (V+1)].
            Data will be overwritten.
        rows: Vocabulary size (including blank token) - V+1.
            Represents the number of threads per block.
        cols: Flattened shape of activation matrix, without vocabulary dimension
            (B * T * U). Represents number of blocks per grid.
        minus: Bool flag whether to add or subtract as reduction.
            If minus is set; calls _reduce_minus, else calls _reduce_rows kernel.
        stream: CUDA Stream.
    """

    return ReduceHelper(
        I_opid=I_Op.IDENTITY.value,
        R_opid=R_Op.MAXIMUM.value,
        acts=acts,
        output=denom,
        num_rows=rows,
        num_cols=cols,
        minus=minus,
        stream=stream,
    )
