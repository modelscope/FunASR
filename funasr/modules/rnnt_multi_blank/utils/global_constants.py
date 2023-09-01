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

import numpy as np
from numba import float32

# Internal globals
_THREADS_PER_BLOCK = 32
_WARP_SIZE = 32
_DTYPE = float32

# Constants
FP32_INF = np.inf
FP32_NEG_INF = -np.inf
THRESHOLD = 1e-1

"""
Getters
"""


def threads_per_block():
    global _THREADS_PER_BLOCK
    return _THREADS_PER_BLOCK


def warp_size():
    global _WARP_SIZE
    return _WARP_SIZE


def dtype():
    global _DTYPE
    return _DTYPE


# RNNT STATUS
class RNNTStatus(enum.Enum):
    RNNT_STATUS_SUCCESS = 0
    RNNT_STATUS_INVALID_VALUE = 1
