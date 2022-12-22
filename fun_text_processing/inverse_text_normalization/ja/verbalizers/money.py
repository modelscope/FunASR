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
from fun_text_processing.inverse_text_normalization.ja.graph_utils import DAMO_CHAR, GraphFst, delete_space
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "12" fractional_part: "05" currency: "$" } -> $12.05

    Args:
        decimal: DecimalFst
    """

    def __init__(self, decimal: GraphFst):
        super().__init__(name="money", kind="verbalize")
        unit = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(DAMO_CHAR - " ", 1)
            + pynutil.delete("\"")
        )
        graph = unit + delete_space + decimal.numbers
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
