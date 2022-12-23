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
from fun_text_processing.text_normalization.zh.graph_utils import FUN_DIGIT, GraphFst, insert_space
from pynini.lib import pynutil


class Fraction(GraphFst):
    '''
        1/5  -> tokens { fraction { numerator: "1" denominator: "5" } }
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        numerator = pynini.closure(FUN_DIGIT, 1) + pynutil.delete('/')
        denominator = pynini.closure(FUN_DIGIT, 1)
        graph = (
            pynutil.insert("numerator: \"")
            + numerator
            + pynutil.insert("\"")
            + insert_space
            + pynutil.insert("denominator: \"")
            + denominator
            + pynutil.insert("\"")
        )

        self.fst = self.add_tokens(graph).optimize()
