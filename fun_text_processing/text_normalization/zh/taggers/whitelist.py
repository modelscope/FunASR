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
from fun_text_processing.text_normalization.zh.graph_utils import GraphFst
from fun_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class Whitelist(GraphFst):
    '''
        ATM  -> tokens { whitelist: "ATM" } 
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        whitelist = pynini.string_file(get_abs_path("data/whitelist/default.tsv"))
        erhua = pynutil.insert("erhua: \"") + pynini.accep('儿') + pynutil.insert("\"")
        sign = pynini.string_file(get_abs_path("data/math/symbol.tsv"))
        whitelist = (
            pynutil.insert("name: \"")
            + (pynini.string_file(get_abs_path("data/erhua/whitelist.tsv")) | whitelist | sign)
            + pynutil.insert("\"")
        )
        graph = pynutil.add_weight(erhua, 0.1) | whitelist

        self.fst = graph.optimize()
