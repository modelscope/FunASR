import pynini
from fun_text_processing.inverse_text_normalization.zh.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.zh.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus twenty three -> cardinal { integer: "23" negative: "-" } }
    Numbers below thirteen are not converted.
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_sign = pynini.string_file(get_abs_path("data/numbers/sign.tsv"))
        graph_dot = pynini.string_file(get_abs_path("data/numbers/dot.tsv"))

        addzero = pynutil.insert("0")
        graph_digits = graph_digit | graph_zero

        # 单位， 十
        # 十一
        graph_teen = pynini.cross("十", "1") + (graph_digit | addzero)

        ##二十一
        graph_tens = graph_digit + pynutil.delete("十") + (graph_digit | addzero)

        # 单位， 百
        # 一百一十 一百零六 一百二十 两百
        graph_hundred = (
            graph_digit
            + pynutil.delete("百")
            + (
                graph_tens
                | graph_teen
                | pynutil.add_weight(graph_zero + graph_digit, 0.1)
                | pynutil.add_weight(graph_digit + addzero, 0.5)
                | pynutil.add_weight(addzero**2, 0.1)
            )
        )

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred @ (
            pynini.closure(DAMO_DIGIT) + (DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT)
        )

        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        # 单位， 千
        # 一千两百三十四 一千零三十四， 一千零四 一千二百
        # 五千
        graph_thousand = (
            (graph_hundred | graph_teen | graph_tens | graph_digits)
            + pynutil.delete("千")
            + (
                graph_hundred
                | pynutil.add_weight(graph_zero + graph_tens, 0.1)
                | pynutil.add_weight(addzero + graph_zero + graph_digit, 0.5)
                | pynutil.add_weight(graph_digit + addzero**2, 0.8)
                | pynutil.add_weight(addzero**3, 1.0)
            )
        )

        # 单位，万
        graph_ten_thousand = (
            (graph_thousand | graph_hundred | graph_teen | graph_tens | graph_digit)
            + pynutil.delete("万")
            + (
                graph_thousand
                | pynutil.add_weight(graph_zero + graph_hundred, 0.1)
                | pynutil.add_weight(addzero + graph_zero + graph_tens, 0.5)
                | pynutil.add_weight(addzero + addzero + graph_zero + graph_digit, 0.5)
                | pynutil.add_weight(graph_digit + addzero**3, 0.8)
                | pynutil.add_weight(addzero**4, 1.0)
            )
        )

        # 单位， 亿/兆
        graph_number = (
            graph_digits
            | graph_teen
            | graph_tens
            | graph_hundred
            | graph_thousand
            | graph_ten_thousand
        )

        graph_number = (
            (graph_number + pynini.accep("兆") + pynutil.delete("零").ques).ques
            + (graph_number + pynini.accep("亿") + pynutil.delete("零").ques).ques
            + graph_number
        )

        ##符号
        graph = graph_number
        # IP
        graph |= graph_digit.plus + (graph_dot + graph_digit.plus).plus

        # just fill
        labels_exception = ["ttttt"]
        graph_exception = pynini.union(*labels_exception)

        self.graph_no_exception = graph

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("负", '"-"'), 0, 1
        )

        final_graph = (
            optional_minus_graph + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
