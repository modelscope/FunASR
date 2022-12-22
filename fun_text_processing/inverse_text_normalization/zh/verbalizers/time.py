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
        time { hours: "12" minutes: "30" } -> 12:30
        time { hours: "1" minutes: "12" } -> 01:12
        time { hours: "2" suffix: "a.m." } -> 02:00 a.m.
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        add_leading_zero_to_double_digit = (DAMO_DIGIT + DAMO_DIGIT) | (pynutil.insert("0") + DAMO_DIGIT)
        
        hour = (
            pynutil.delete("hour:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(DAMO_DIGIT, 1)
            + pynutil.delete("\"")
        )
        minute = (
            pynutil.delete("minute:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(DAMO_DIGIT, 1)
            + pynutil.delete("\"")
        )

        second = (
            pynutil.delete("second:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(DAMO_DIGIT, 1)
            + pynutil.delete("\"")
        )

        suffix = (
            delete_space
            + pynutil.insert(" ")
            + pynutil.delete("noon:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(DAMO_CHAR - " ", 1)
            + pynutil.delete("\"")
        )
        optional_suffix = pynini.closure(suffix, 0, 1)
        zone = (
            delete_space
            + pynutil.insert(" ")
            + pynutil.delete("zone:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(DAMO_CHAR - " ", 1)
            + pynutil.delete("\"")
        )
        optional_zone = pynini.closure(zone, 0, 1)
        graph = (
            hour @ add_leading_zero_to_double_digit
            + delete_space
            + pynutil.insert(":")
            + (minute @ add_leading_zero_to_double_digit)
            + (pynutil.insert(":") + second @ add_leading_zero_to_double_digit).ques
            + optional_suffix
            + optional_zone
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
