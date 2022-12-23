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
from fun_text_processing.inverse_text_normalization.vi.graph_utils import (
    DAMO_CHAR,
    DAMO_DIGIT,
    GraphFst,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "3" } -> 3h
        time { hours: "12" minutes: "30" } -> 12:30
        time { hours: "1" minutes: "12" second: "22"} -> 1:12:22
        time { minutes: "36" second: "45"} -> 36p45s
        time { hours: "2" zone: "gmt" } -> 2h gmt
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        add_leading_zero_to_double_digit = (DAMO_DIGIT + DAMO_DIGIT) | (pynutil.insert("0") + DAMO_DIGIT)
        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_DIGIT, 1)
            + pynutil.delete('"')
        )
        minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_DIGIT, 1)
            + pynutil.delete('"')
        )
        second = (
            pynutil.delete("seconds:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_DIGIT, 1)
            + pynutil.delete('"')
        )
        zone = (
            delete_space
            + insert_space
            + pynutil.delete("zone:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_CHAR - " ", 1)
            + pynutil.delete('"')
        )
        optional_zone = pynini.closure(zone, 0, 1)
        optional_second = pynini.closure(
            delete_space + pynutil.insert(":") + (second @ add_leading_zero_to_double_digit), 0, 1,
        )

        graph_h = hour + pynutil.insert("h")
        graph_hms = (
            hour + delete_space + pynutil.insert(":") + (minute @ add_leading_zero_to_double_digit) + optional_second
        )
        graph_ms = (
            minute
            + delete_space
            + pynutil.insert("p")
            + (second @ add_leading_zero_to_double_digit)
            + pynutil.insert("s")
        )

        graph = (graph_h | graph_ms | graph_hms) + optional_zone
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
