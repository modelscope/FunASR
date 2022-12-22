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
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        tokens { decimal { integer_part: "одно целая" fractional_part: "восемь сотых} } ->
            "одно целая восемь сотых"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.closure(pynini.cross("negative: \"true\" ", "минус "), 0, 1)
        integer = pynutil.delete(" \"") + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        integer_part = pynutil.delete("integer_part:") + integer
        fractional_part = pynutil.delete("fractional_part:") + integer
        optional_quantity_part = pynini.closure(
            pynini.accep(" ")
            + pynutil.delete("quantity: \"")
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete("\""),
            0,
            1,
        )

        self.graph = (
            optional_sign + integer_part + pynini.accep(" ") + fractional_part + optional_quantity_part + delete_space
        )
        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
