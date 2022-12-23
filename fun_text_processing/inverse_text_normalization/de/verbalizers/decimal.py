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
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst, delete_preserve_order
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "true" integer_part: "12"  fractional_part: "5006" quantity: "billion" } -> -12.5006 billion

    Args:
        tn_decimal_verbalizer: TN decimal verbalizer
    """

    def __init__(self, tn_decimal_verbalizer: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)
        delete_space = pynutil.delete(" ")
        optional_sign = pynini.closure(
            pynutil.delete("negative: \"") + DAMO_NOT_QUOTE + pynutil.delete("\"") + delete_space, 0, 1
        )
        optional_integer = pynini.closure(tn_decimal_verbalizer.integer, 0, 1)
        optional_fractional = pynini.closure(
            delete_space + pynutil.insert(",") + tn_decimal_verbalizer.fractional_default, 0, 1
        )
        graph = (optional_integer + optional_fractional + tn_decimal_verbalizer.optional_quantity).optimize()
        self.numbers = optional_sign + graph
        graph = self.numbers + delete_preserve_order
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
