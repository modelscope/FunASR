import pynini
from fun_text_processing.text_normalization.zh.graph_utils import (
    FUN_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from fun_text_processing.text_normalization.zh.utils import UNIT_1e01, get_abs_path
from pynini.lib import pynutil


class Time(GraphFst):
    """
    tokens { time { h: "1" m: "02" s: "36" } } -> 一点零二分三十六秒
    tokens { time { suffix "am"  hours: "1" minutes: "02" seconds: "36" } } -> 上午一点零二分三十六秒
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/number/digit_teen.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        graph_no_zero = pynini.cross("0", "")

        graph_digit_no_zero = graph_digit | graph_no_zero

        graph_2_digit_zero_none = pynini.cross("0", "") + pynini.cross("0", "")
        graph_2_digit_zero = pynini.cross("00", "零")

        graph_2_digit_time = (graph_teen + pynutil.insert(UNIT_1e01) + graph_digit_no_zero) | (
            graph_zero + graph_digit
        )
        h = graph_2_digit_time | graph_2_digit_zero | graph_digit
        m = graph_2_digit_time | graph_2_digit_zero
        s = graph_2_digit_time | graph_2_digit_zero

        # 6:25
        h_m = (
            pynutil.delete('hours: "')
            + h
            + pynutil.insert("点")
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete('minutes: "')
            + (graph_2_digit_time)
            + pynutil.insert("分")
            + pynutil.delete('"')
        )

        # 23:00
        h_00 = (
            pynutil.delete('hours: "')
            + h
            + pynutil.insert("点")
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete('minutes: "')
            + (graph_2_digit_zero_none)
            + pynutil.delete('"')
        )

        # 9:12:52
        h_m_s = (
            pynutil.delete('hours: "')
            + h
            + pynutil.insert("点")
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete('minutes: "')
            + m
            + pynutil.insert("分")
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete('seconds: "')
            + s
            + pynutil.insert("秒")
            + pynutil.delete('"')
        )

        graph = h_m | h_m_s | h_00
        graph_suffix = (
            pynutil.delete('suffix: "')
            + pynini.closure(FUN_NOT_QUOTE)
            + pynutil.delete('"')
            + delete_space
            + graph
        )
        graph |= graph_suffix
        self.fst = self.delete_tokens(graph).optimize()
