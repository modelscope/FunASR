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

from fun_text_processing.text_normalization.zh.graph_utils import GraphFst
from fun_text_processing.text_normalization.zh.taggers.cardinal import Cardinal
from pynini.lib import pynutil


class Fraction(GraphFst):
    '''
        tokens { fraction { denominator: "5" numerator: "1" } } -> 五分之一      
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        denominator = pynutil.delete("denominator: \"") + Cardinal().graph_cardinal + pynutil.delete("\"")
        numerator = pynutil.delete("numerator: \"") + Cardinal().graph_cardinal + pynutil.delete("\"")
        graph = denominator + pynutil.delete(" ") + pynutil.insert("分之") + numerator

        self.fst = self.delete_tokens(graph).optimize()
