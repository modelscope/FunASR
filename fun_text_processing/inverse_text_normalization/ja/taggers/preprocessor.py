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
from fun_text_processing.inverse_text_normalization.ja.graph_utils import DAMO_SIGMA, GraphFst
from fun_text_processing.inverse_text_normalization.ja.utils import get_abs_path
from pynini.lib import pynutil


class PreProcessor(GraphFst):
    def __init__(
        self, halfwidth_to_fullwidth: bool = True,
    ):
        super().__init__(name="PreProcessor", kind="processor")

        graph = pynini.cdrewrite('', '', '', DAMO_SIGMA)

        if halfwidth_to_fullwidth:
            halfwidth_to_fullwidth_graph = pynini.string_file(get_abs_path('data/char/halfwidth_to_fullwidth.tsv'))
            graph @= pynini.cdrewrite(halfwidth_to_fullwidth_graph, '', '', DAMO_SIGMA)

        self.fst = graph.optimize()
