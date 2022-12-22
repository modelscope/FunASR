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
from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: email addresses, etc.
        e.g. c d f eins at a b c punkt e d u -> tokens { name: "cdf1.abc.edu" }
    
    Args:
        tn_electronic_tagger: TN eletronic tagger
        tn_electronic_verbalizer: TN eletronic verbalizer
    """

    def __init__(self, tn_electronic_tagger: GraphFst, tn_electronic_verbalizer: GraphFst, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        tagger = pynini.invert(tn_electronic_verbalizer.graph).optimize()
        verbalizer = pynini.invert(tn_electronic_tagger.graph).optimize()
        final_graph = tagger @ verbalizer

        graph = pynutil.insert("name: \"") + final_graph + pynutil.insert("\"")
        self.fst = graph.optimize()
