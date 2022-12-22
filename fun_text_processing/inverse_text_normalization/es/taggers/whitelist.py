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
from fun_text_processing.inverse_text_normalization.es.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space
from pynini.lib import pynutil


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelisted tokens
        e.g. usted -> tokens { name: "ud." }
    This class has highest priority among all classifier grammars. Whitelisted tokens are defined and loaded from "data/whitelist.tsv".
    """

    def __init__(self):
        super().__init__(name="whitelist", kind="classify")

        whitelist = pynini.string_file(get_abs_path("data/whitelist.tsv")).invert()
        graph = pynutil.insert("name: \"") + convert_space(whitelist) + pynutil.insert("\"")
        self.fst = graph.optimize()
