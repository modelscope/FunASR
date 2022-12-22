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
from fun_text_processing.inverse_text_normalization.zh.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { month: "january" day: "5" year: "2012" preserve_order: true } -> february 5 2012
        date { day: "5" month: "january" year: "2012" preserve_order: true } -> 5 february 2012
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete("\"")
        )

        graph_ymd = (
            year + pynini.cross(" ", "年") + month + pynini.cross(" ", "月") + day + pynutil.insert("日")
        )

        graph_md = (
            month + pynini.cross(" ", "月") + day + pynutil.insert("日")
        )

        graph_ym = (
            year + pynini.cross(" ", "年") + month + pynutil.insert("月")
        )

        graph_year = (
            year + pynutil.insert("年")
        )
        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete("field_order:")
            + delete_space
            + pynutil.delete("\"")
            + DAMO_NOT_QUOTE
            + pynutil.delete("\"")
            + delete_space
        )

        final_graph = (graph_ymd | graph_ym | graph_md | graph_year) + delete_space + optional_preserve_order

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
