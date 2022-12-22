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


class Measure(GraphFst):
    '''
        tokens { measure { cardinal: "一" } units: "千克" } } ->  一千克
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        graph = (
            pynutil.delete("cardinal {")
            + delete_space
            + pynutil.delete("integer: \"")
            + pynini.closure(FUN_NOT_QUOTE)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("}")
            + delete_space
            + pynutil.delete("units: \"")
            + pynini.closure(FUN_NOT_QUOTE)
            + pynutil.delete("\"")
        )
        percent_graph = (
            pynutil.delete("decimal { ")
            + pynutil.delete("integer_part: \"")
            + pynutil.insert("百分之")
            + pynini.closure(FUN_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("}")
        )
        graph |= percent_graph
        self.fst = self.delete_tokens(graph).optimize()
