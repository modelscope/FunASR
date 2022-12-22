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
from fun_text_processing.inverse_text_normalization.vi.graph_utils import GraphFst, delete_space
from fun_text_processing.inverse_text_normalization.vi.utils import get_abs_path
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. thứ nhất -> ordinal { integer: "1" }
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="classify")

        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_ordinal = pynini.cross("thứ", "")
        graph = graph_digit

        self.graph = graph
        final_graph = pynutil.insert('integer: "') + graph_ordinal + delete_space + self.graph + pynutil.insert('"')
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
