import pynini
from fun_text_processing.inverse_text_normalization.tl.utils import get_abs_path, num_to_word
from fun_text_processing.inverse_text_normalization.tl.graph_utils import (
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

        graph_digit_ties_all = pynini.string_file(get_abs_path("data/numbers/digit_ties.tsv"))

        graph_digit_ties1 = pynini.string_file(get_abs_path("data/numbers/digit_ties1.tsv"))
        graph_digit_ties2 = pynini.string_file(get_abs_path("data/numbers/digit_ties2.tsv"))

        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))

        graph_teens_using_ties = graph_ties + pynini.cross(" ", "") + graph_digit

        addzero = pynutil.insert("0")
        zero = graph_zero

        ##一位数
        graph_digits = graph_digit | graph_zero
        digits = graph_digits
        digit = graph_digit
        digit_ties = graph_digit_ties1 | graph_digit_ties2

        ##两位数
        graph_teens = graph_teen | graph_teens_using_ties
        teens = graph_teens

        ##三位数, daan 百，raan 百（只有4和9的时候）
        graph_hundred1 = pynutil.delete("daan")
        graph_hundred2 = pynutil.delete("raan")
        graph_at = pynutil.delete("at")

        delete_at = graph_at

        graph_hundred_component1 = graph_digit_ties1 + delete_space + graph_hundred1
        graph_hundred_component2 = graph_digit_ties2 + delete_space + graph_hundred2

        graph_hundred_component = graph_hundred_component1 | graph_hundred_component2

        hundred = (graph_hundred_component + pynutil.insert("00")) | (
            graph_hundred_component
            + delete_space
            + delete_at
            + delete_space
            + (
                teens
                | pynutil.add_weight(addzero + digit, 0.1)
                | pynutil.add_weight(digit + addzero, 0.5)
            )
        )
        graph_hundred_component_at_least_one_none_zero_digit = hundred @ (
            pynini.closure(DAMO_DIGIT) + (DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        ##千， libo 表示千
        thousand = (
            (hundred | teens | digit_ties)
            + delete_space
            + pynutil.delete("libo")
            + delete_space
            + (
                hundred
                | (delete_at + delete_space).ques + pynutil.add_weight(addzero + teens, 0.1)
                | (delete_at + delete_space).ques + pynutil.add_weight(addzero**2 + digit, 0.5)
                | pynutil.add_weight(digit + addzero**2, 0.8)
                | pynutil.add_weight(addzero**3, 1.0)
            )
        )

        ##百万，milyon表示百万
        # million = (((hundred | teens | digit_ties) + delete_space + pynutil.delete("milyon") | pynutil.insert("000", weight=0.1))+ delete_space + (
        #            thousand
        #            | pynutil.add_weight(addzero + hundred, 0.1)
        #            | (delete_at + delete_space).ques + pynutil.add_weight(addzero**2 + teens, 0.5)
        #            | (delete_at + delete_space).ques + pynutil.add_weight(addzero + addzero + addzero + digit, 0.5)
        #            | pynutil.add_weight(digit + addzero**3, 0.8)
        #            | pynutil.add_weight(addzero**4, 1.0)))

        graph_million = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("milyon"),
            pynutil.insert("000", weight=0.1),
        )

        ##bilyon bilyon表示十亿
        graph_billion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("bilyon"),
            pynutil.insert("000", weight=0.1),
        )

        ##trilyon trilyon表示兆
        graph_trillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("trilyon"),
            pynutil.insert("000", weight=0.1),
        )

        ##
        graph_quadrillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("quadrilyon"),
            pynutil.insert("000", weight=0.1),
        )
        graph_quintillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("quintilyon"),
            pynutil.insert("000", weight=0.1),
        )
        graph_sextillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("sextilyon"),
            pynutil.insert("000", weight=0.1),
        )

        #
        graph = pynini.union(
            graph_sextillion
            + delete_space
            + graph_quintillion
            + delete_space
            + graph_quadrillion
            + delete_space
            + graph_trillion
            + delete_space
            + graph_billion
            + delete_space
            + graph_million
            + delete_space
            + thousand
            + delete_space
            + graph_hundred_component,
            thousand,
            hundred,
            teens,
            digits,
            graph_zero,
        )

        # graph = zero | digits | teens | hundred | thousand | million

        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0"))
            + pynini.difference(DAMO_DIGIT, "0")
            + pynini.closure(DAMO_DIGIT),
            "0",
        )

        labels_exception = [num_to_word(x) for x in range(0, 13)]
        graph_exception = pynini.union(*labels_exception)

        # graph = (
        #    pynini.cdrewrite(pynutil.delete("and"), DAMO_SPACE, DAMO_SPACE, DAMO_SIGMA)
        #    @ (DAMO_ALPHA + DAMO_SIGMA)
        #    @ graph
        # )

        self.graph_no_exception = graph

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", '"-"') + DAMO_SPACE, 0, 1
        )

        final_graph = (
            optional_minus_graph + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
