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
from fun_text_processing.text_normalization.zh.graph_utils import FUN_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class Whitelist(GraphFst):
    '''
        tokens { whitelist: "ATM" } -> A T M
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="whitelist", kind="verbalize", deterministic=deterministic)
        remove_erhua = pynutil.delete("erhua: \"") + pynutil.delete("儿") + pynutil.delete("\"")
        whitelist = pynutil.delete("name: \"") + pynini.closure(FUN_NOT_QUOTE) + pynutil.delete("\"")
        graph = remove_erhua | whitelist
        self.fst = graph.optimize()
