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
from fun_text_processing.text_normalization.zh.graph_utils import GraphFst, insert_space
from fun_text_processing.text_normalization.zh.taggers.cardinal import Cardinal
from fun_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class Measure(GraphFst):
    '''
        1kg  -> tokens { measure { cardinal { integer: "一" } units: "千克" } }
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        units_en = pynini.string_file(get_abs_path("data/measure/units_en.tsv"))
        units_zh = pynini.string_file(get_abs_path("data/measure/units_zh.tsv"))
        graph = (
            pynutil.insert("cardinal { ")
            + pynutil.insert("integer: \"")
            + Cardinal().graph_cardinal
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + insert_space
            + pynutil.insert("units: \"")
            + (units_en | units_zh)
            + pynutil.insert("\"")
        )
        percent_graph = (
            pynutil.insert("decimal { ")
            + pynutil.insert("integer_part: \"")
            + Cardinal().graph_cardinal
            + pynutil.delete("%")
            + pynutil.insert("\"")
            + pynutil.insert(" }")
        )
        graph |= percent_graph

        self.fst = self.add_tokens(graph).optimize()
