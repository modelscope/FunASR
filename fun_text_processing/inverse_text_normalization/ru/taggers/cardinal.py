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
from fun_text_processing.text_normalization.en.graph_utils import DAMO_DIGIT, GraphFst, insert_space
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
       "тысяча один" ->  cardinal { integer: "1 001" }

    Args:
        tn_cardinal: Text normalization Cardinal graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        graph = tn_cardinal.cardinal_numbers_default
        self.graph = graph.invert().optimize()

        optional_sign = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("минус ", "\"-\"") + insert_space, 0, 1
        )

        # do not invert numbers less than 10
        graph = pynini.compose(graph, DAMO_DIGIT ** (2, ...))
        graph = optional_sign + pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
