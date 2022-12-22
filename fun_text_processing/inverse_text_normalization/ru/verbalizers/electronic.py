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
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. electronic { username: "ab@nd.ru" } -> "ab@nd.ru"
    """

    def __init__(self):
        super().__init__(name="electronic", kind="verbalize")

        graph = pynutil.delete("username: \"") + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
