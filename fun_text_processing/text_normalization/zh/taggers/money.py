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


class Money(GraphFst):
    '''
        ￥1.25 -> tokens { money { fractional_part: "元" integer_part: "一点五" } }
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        currency_code = pynini.string_file(get_abs_path("data/money/currency_code.tsv"))
        currency_symbol = pynini.string_file(get_abs_path("data/money/currency_symbol.tsv"))
        graph = (
            pynutil.insert("fractional_part: \"")
            + (currency_code | currency_symbol)
            + pynutil.insert("\"")
            + insert_space
            + pynutil.insert("integer_part: \"")
            + Cardinal().graph_cardinal
            + pynutil.insert("\"")
        )

        self.fst = self.add_tokens(graph).optimize()
