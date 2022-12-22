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
from fun_text_processing.text_normalization.zh.graph_utils import FUN_NOT_QUOTE, GraphFst, delete_space
from fun_text_processing.text_normalization.zh.utils import UNIT_1e01, get_abs_path
from pynini.lib import pynutil


class Date(GraphFst):
    '''
        tokens { date { year: "2002" month: "01" day: "28"} }  ->  二零零二年一月二十八日
        tokens { date { year: "2002" } } ->  二零零八年
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)
        date_type0 = pynutil.delete('year: \"') + pynini.closure(FUN_NOT_QUOTE) + pynutil.delete('\"')
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/number/digit_teen.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        graph_no_zero = pynini.cross("0", "")
        graph_year = pynini.closure(graph_digit | graph_zero, 2, 4)
        graph_digit_no_zero = graph_digit | graph_no_zero
        graph_2_digit_date = (graph_teen + pynutil.insert(UNIT_1e01) + graph_digit_no_zero) | (
            graph_no_zero + graph_digit
        )

        date_type1 = (
            pynutil.delete("year: \"")
            + graph_year
            + pynutil.insert("年")
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("month: \"")
            + graph_2_digit_date
            + pynutil.insert("月")
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("day: \"")
            + graph_2_digit_date
            + pynutil.insert("日")
            + pynutil.delete("\"")
        )

        date_type2 = (
            pynutil.delete("year: \"")
            + graph_year
            + pynutil.insert("年")
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("month: \"")
            + graph_2_digit_date
            + pynutil.insert("月")
            + pynutil.delete("\"")
        )

        graph = date_type0 | date_type1 | date_type2

        self.fst = self.delete_tokens(graph).optimize()
