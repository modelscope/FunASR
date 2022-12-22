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

from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone, e.g.
        "восемь девятьсот тринадцать девятьсот восемьдесят три пятьдесят шесть ноль один" -> telephone { number_part: "8-913-983-56-01" }

    Args:
        tn_telephone: Text normalization telephone graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_telephone: GraphFst, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        tn_telephone = tn_telephone.final_graph
        graph = tn_telephone.invert().optimize()
        graph = pynutil.insert("number_part: \"") + graph + pynutil.insert("\"")
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
