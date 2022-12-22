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
from fun_text_processing.inverse_text_normalization.vi.graph_utils import DAMO_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "true" integer_part: "12"  fractional_part: "5006" quantity: "tỷ" } -> -12.5006 tỷ
    """

    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")
        optionl_sign = pynini.closure(pynini.cross('negative: "true"', "-") + delete_space, 0, 1)
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_integer = pynini.closure(integer + delete_space, 0, 1)
        fractional = (
            pynutil.insert(".")
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_fractional = pynini.closure(fractional + delete_space, 0, 1)
        quantity = (
            pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_quantity = pynini.closure(pynutil.insert(" ") + quantity + delete_space, 0, 1)
        graph = optional_integer + optional_fractional + optional_quantity
        self.numbers = graph
        graph = optionl_sign + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
