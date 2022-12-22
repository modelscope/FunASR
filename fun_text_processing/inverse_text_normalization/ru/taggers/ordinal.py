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
    Finite state transducer for classifying ordinals, e.g.
        "второе" -> ordinal { integer: "2" } }

    Args:
        tn_ordinal: Text normalization Ordinal graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        tn_ordinal = tn_ordinal.ordinal_numbers

        graph = tn_ordinal.invert().optimize()
        self.graph = graph

        # do not invert numbers less than 10
        graph = pynini.compose(graph, DAMO_DIGIT ** (2, ...))
        graph = pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
