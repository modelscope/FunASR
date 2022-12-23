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
from fun_text_processing.text_normalization.en.graph_utils import DAMO_DIGIT, GraphFst
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal, e.g.
        13th -> ordinal { integer: "thirteen" }
        
    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph
        cardinal_format = pynini.closure(DAMO_DIGIT | pynini.accep(","))
        st_format = (
            pynini.closure(cardinal_format + (DAMO_DIGIT - "1"), 0, 1)
            + pynini.accep("1")
            + pynutil.delete(pynini.union("st", "ST"))
        )
        nd_format = (
            pynini.closure(cardinal_format + (DAMO_DIGIT - "1"), 0, 1)
            + pynini.accep("2")
            + pynutil.delete(pynini.union("nd", "ND"))
        )
        rd_format = (
            pynini.closure(cardinal_format + (DAMO_DIGIT - "1"), 0, 1)
            + pynini.accep("3")
            + pynutil.delete(pynini.union("rd", "RD"))
        )
        th_format = pynini.closure(
            (DAMO_DIGIT - "1" - "2" - "3")
            | (cardinal_format + "1" + DAMO_DIGIT)
            | (cardinal_format + (DAMO_DIGIT - "1") + (DAMO_DIGIT - "1" - "2" - "3")),
            1,
        ) + pynutil.delete(pynini.union("th", "TH"))
        self.graph = (st_format | nd_format | rd_format | th_format) @ cardinal_graph
        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
