# Copyright NeMo (https://github.com/NVIDIA/NeMo). All Rights Reserved.
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

import pynini
from fun_text_processing.text_normalization.zh.graph_utils import FUN_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import pynutil


class Money(GraphFst):
    '''
        tokens { money { integer_part: "一点五" fractional_part: "元" } } ->  一点五元
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        cur = pynutil.delete("fractional_part: \"") + pynini.closure(FUN_NOT_QUOTE) + pynutil.delete("\"")
        num = pynutil.delete("integer_part: \"") + pynini.closure(FUN_NOT_QUOTE) + pynutil.delete("\"") + delete_space
        graph = num + cur

        self.fst = self.delete_tokens(graph).optimize()
