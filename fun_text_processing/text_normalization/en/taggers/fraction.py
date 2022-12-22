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
from fun_text_processing.text_normalization.en.graph_utils import GraphFst, get_abs_path
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "23 4/5" ->
    tokens { fraction { integer: "twenty three" numerator: "four" denominator: "five" } }
    "23 4/5th" ->
    tokens { fraction { integer: "twenty three" numerator: "four" denominator: "five" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph

        integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        numerator = (
            pynutil.insert("numerator: \"") + cardinal_graph + (pynini.cross("/", "\" ") | pynini.cross(" / ", "\" "))
        )

        endings = ["rd", "th", "st", "nd"]
        endings += [x.upper() for x in endings]
        optional_end = pynini.closure(pynini.cross(pynini.union(*endings), ""), 0, 1)

        denominator = pynutil.insert("denominator: \"") + cardinal_graph + optional_end + pynutil.insert("\"")

        graph = pynini.closure(integer + pynini.accep(" "), 0, 1) + (numerator + denominator)
        graph |= pynini.closure(integer + (pynini.accep(" ") | pynutil.insert(" ")), 0, 1) + pynini.compose(
            pynini.string_file(get_abs_path("data/number/fraction.tsv")), (numerator + denominator)
        )

        self.graph = graph
        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
