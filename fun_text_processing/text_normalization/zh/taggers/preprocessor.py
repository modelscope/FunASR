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
from fun_text_processing.text_normalization.zh.graph_utils import FUN_SIGMA, GraphFst
from fun_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class PreProcessor(GraphFst):
    '''
        Preprocessing of TN:
            1. interjections removal such as '啊, 呃'
            2. fullwidth -> halfwidth char conversion
        好啊 -> 好
        呃对 -> 对
        ：   -> :
        ；   -> ;
    '''

    def __init__(
        self, remove_interjections: bool = True, fullwidth_to_halfwidth: bool = True,
    ):
        super().__init__(name="PreProcessor", kind="processor")

        graph = pynini.cdrewrite('', '', '', FUN_SIGMA)

        if remove_interjections:
            remove_interjections_graph = pynutil.delete(pynini.string_file(get_abs_path('data/denylist/denylist.tsv')))
            graph @= pynini.cdrewrite(remove_interjections_graph, '', '', FUN_SIGMA)

        if fullwidth_to_halfwidth:
            fullwidth_to_halfwidth_graph = pynini.string_file(get_abs_path('data/char/fullwidth_to_halfwidth.tsv'))
            graph @= pynini.cdrewrite(fullwidth_to_halfwidth_graph, '', '', FUN_SIGMA)

        self.fst = graph.optimize()
