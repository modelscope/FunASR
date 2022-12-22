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

from fun_text_processing.inverse_text_normalization.ko.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    convert_space,
    delete_space,
    delete_extra_space,
    DAMO_SIGMA,
    DAMO_CHAR,
    DAMO_SPACE,
)
import pynini
from pynini.lib import pynutil

class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")
        # integer_part # numerator # denominator
        
        graph_cardinal = cardinal.graph_no_exception
       
        #without the integerate part 
        #分子
        numerator = pynutil.insert('numerator: "') + graph_cardinal + pynutil.insert('"')
        #分母
        denominator = pynutil.insert('denominator: "') + graph_cardinal + pynutil.delete("분의") + pynutil.insert('"')

        ##
        graph_fraction_component = denominator + pynini.cross(" ", " ") + numerator

        self.graph_fraction_component = graph_fraction_component

        graph = graph_fraction_component
        graph = graph.optimize()
        self.final_graph_wo_negative = graph

        ##负
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("마이너스", "\"true\"") + DAMO_SPACE, 0, 1
        )

        graph = optional_graph_negative + graph
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize() 
