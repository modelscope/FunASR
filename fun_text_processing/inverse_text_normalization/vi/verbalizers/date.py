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
from fun_text_processing.inverse_text_normalization.vi.graph_utils import DAMO_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { month: "1" year: "2012"} -> tháng 1 năm 2012
        date { day: "5" month: "10" year: "2021" preserve_order: true } -> 5 tháng 10 năm 2021
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")
        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete('"')
        )

        # (day) month year
        # day month
        graph_dm = day + delete_space + pynutil.insert(" tháng ") + month
        graph_dmy = graph_dm + delete_space + pynutil.insert(" năm ") + year
        graph_m = pynutil.insert("tháng ") + month
        graph_my = pynutil.insert("tháng ") + month + delete_space + pynutil.insert(" năm ") + year
        graph_y = pynutil.insert("năm ") + year

        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete("field_order:")
            + delete_space
            + pynutil.delete('"')
            + DAMO_NOT_QUOTE
            + pynutil.delete('"')
            + delete_space
        )

        final_graph = (graph_y | graph_m | graph_dm | graph_dmy | graph_my) + delete_space + optional_preserve_order

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
