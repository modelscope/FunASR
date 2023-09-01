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


import math
from typing import Optional, Tuple

import torch
from numba import cuda

from funasr.modules.rnnt_multi_blank.utils import global_constants

threshold = global_constants.THRESHOLD


@cuda.jit(device=True, inline=True)
def log_sum_exp(a: float, b: float):
    if a == global_constants.FP32_NEG_INF:
        return b

    if b == global_constants.FP32_NEG_INF:
        return a

    if a > b:
        return math.log1p(math.exp(b - a)) + a
    else:
        return math.log1p(math.exp(a - b)) + b


@cuda.jit(device=True, inline=True)
def div_up(x: int, y: int):
    return (x + y - 1) // y


@cuda.jit(device=True)
def maximum(x, y):
    if x < y:
        return y
    else:
        return x


@cuda.jit(device=True)
def add(x, y):
    return x + y


@cuda.jit(device=True)
def identity(x):
    return x


@cuda.jit(device=True)
def negate(x):
    return -x


@cuda.jit(device=True)
def exponential(x):
    return math.exp(x)


@cuda.jit(device=True)
def log_plus(p1: float, p2: float):
    if p1 == global_constants.FP32_NEG_INF:
        return p2

    if p2 == global_constants.FP32_NEG_INF:
        return p1

    result = math.log1p(math.exp(-math.fabs(p1 - p2))) + maximum(p1, p2)
    return result


@cuda.jit(device=True, inline=True)
def copy_data_1d(source: torch.Tensor, dest: torch.Tensor, idx: int):
    dest[idx] = source[idx]


@cuda.jit()
def compute_costs_data(
    source: torch.Tensor, dest: torch.Tensor, fastemit_lambda: float
):
    block = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    idx = block * cuda.blockDim.x + tid
    length = source.shape[0]

    if idx < length:
        copy_data_1d(source, dest, idx)
        dest[idx] *= -1.0
        dest[idx] *= 1.0 + fastemit_lambda


def get_workspace_size(
    maxT: int, maxU: int, minibatch: int, gpu: bool
) -> Tuple[Optional[int], global_constants.RNNTStatus]:
    if minibatch <= 0 or maxT <= 0 or maxU <= 0:
        return (None, global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE)

    # per minibatch memory
    per_minibatch_size = 0

    # alphas & betas
    per_minibatch_size += maxT * maxU * 2

    if not gpu:
        # // blank & label log probability cache
        per_minibatch_size += maxT * maxU * 2
    else:
        # // softmax denominator
        per_minibatch_size += maxT * maxU
        # // forward - backward loglikelihood
        per_minibatch_size += 2

    size = per_minibatch_size * minibatch
    return (size, global_constants.RNNTStatus.RNNT_STATUS_SUCCESS)


def flatten_tensor(x: torch.Tensor):
    original_shape = x.shape
    x = x.view([-1])
    return x, original_shape
