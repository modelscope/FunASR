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
from fun_text_processing.text_normalization.zh.graph_utils import GraphFst, insert_space
from fun_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class Time(GraphFst):
    '''
        1:02    -> tokens { time { hours: "1" minitus: "02" } }
        1:02:36 -> tokens { time { hours: "1" minutes: "02" seconds: "36" } }
        1:02 am -> tokens { time { hours: "1" minutes: "02" seconds: "36" suffix "am" } }
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        h = pynini.string_file(get_abs_path("data/time/hour.tsv"))
        time_tens = pynini.string_file(get_abs_path("data/time/tens.tsv"))
        time_digit = pynini.string_file(get_abs_path("data/time/digit.tsv"))
        time_zero = pynini.string_file(get_abs_path("data/time/zero.tsv"))
        time_digit = time_digit | time_zero
        time_suffix = pynini.string_file(get_abs_path("data/time/suffix.tsv"))

        m = time_tens + time_digit
        s = (time_tens + time_digit) | time_digit

        delete_colon = pynini.cross(':', ' ')

        # 5:05, 14:30
        h_m = (
            pynutil.insert('hours: \"')
            + h
            + pynutil.insert('\"')
            + delete_colon
            + pynutil.insert('minutes: \"')
            + m
            + pynutil.insert('\"')
        )

        # 1:30:15
        h_m_s = (
            pynutil.insert('hours: \"')
            + h
            + pynutil.insert('\"')
            + delete_colon
            + pynutil.insert('minutes: \"')
            + m
            + pynutil.insert('\"')
            + delete_colon
            + pynutil.insert('seconds: \"')
            + s
            + pynutil.insert('\"')
        )

        graph = h_m | h_m_s
        graph_suffix = graph + insert_space + pynutil.insert('suffix: \"') + time_suffix + pynutil.insert('\"')
        graph |= graph_suffix
        self.fst = self.add_tokens(graph).optimize()
