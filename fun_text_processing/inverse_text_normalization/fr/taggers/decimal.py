import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    DAMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_hyphen,
    delete_space,
)
from fun_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


def get_quantity(
    decimal: "pynini.FstLike", cardinal_up_to_thousand: "pynini.FstLike"
) -> "pynini.FstLike":
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. one million -> integer_part: "1" quantity: "million"
    e.g. one point five million -> integer_part: "1" fractional_part: "5" quantity: "million"

    Will tag cases up to denominations of tens of hundreds of thousand. 'douze cent mille millions' -> 1 200 000 millions

    Args:
        decimal: decimal FST
        cardinal_up_to_million: cardinal FST
    """
    numbers = cardinal_up_to_thousand @ (
        pynutil.delete(pynini.closure("0"))
        + pynini.difference(DAMO_DIGIT, "0")
        + pynini.closure(DAMO_DIGIT)
    )

    suffix = pynini.union(
        "million",
        "millions",
        "milliard",
        "milliards",
        "billion",
        "billions",
        "billiard",
        "billiards",
        "trillion",
        "trillions",
        "trilliard",
        "trilliards",
    )
    res = (
        pynutil.insert('integer_part: "')
        + numbers
        + pynutil.insert('"')
        + (
            pynini.union(delete_hyphen, delete_extra_space)
        )  # Can be written either as 'deux-millions' or 'deux millions' depending on whether it registers as a noun or part of cardinal.
        + pynutil.insert(' quantity: "')
        + suffix
        + pynutil.insert('"')
    )
    res |= (
        decimal + delete_extra_space + pynutil.insert(' quantity: "') + suffix + pynutil.insert('"')
    )
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        Decimal point is "," (virgule).
            e.g. moins un virgule deux six -> decimal { negative: "true" integer_part: "1" fractional_part: "26" }

        This decimal rule assumes that decimals can be pronounced as:
        (a cardinal) + ('virgule') plus (any sequence of cardinals <1 million, including 'zero')

        Also writes large numbers in shortened form, e.g.
            e.g. un virgule deux-six-million -> decimal { negative: "false" integer_part: "1" fractional_part: "26" quantity: "million" }
            e.g. deux-million -> decimal { negative: "false" integer_part: "2" quantity: "millions" }
            e.g. moins cent-vingt-quatre-millions -> decimal { negative: "true" integer_part: "124" quantity: "millions" }
    Args:
        cardinal: CardinalFst

    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        # number after decimal point can be any series of cardinals <1 million, including 'zero'
        graph_decimal = cardinal.numbers_up_to_million
        graph_decimal = pynini.closure(graph_decimal + delete_space) + graph_decimal
        self.graph = graph_decimal

        # decimal point is denote by virgule
        graph_fractional_separator = pynutil.delete("virgule")

        # Possible negatives
        optional_graph_negative = (
            pynutil.insert("negative: ") + pynini.cross("moins", '"true"') + delete_extra_space
        )
        optional_graph_negative = optional_graph_negative.ques

        # Fractional portion
        graph_fractional = (
            pynutil.insert('fractional_part: "') + graph_decimal + pynutil.insert('"')
        )

        # Integers
        cardinal_graph = cardinal.graph_no_exception | pynini.string_file(
            get_abs_path("data/numbers/zero.tsv")
        )
        graph_integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')

        # Final graphs
        final_graph_wo_sign = (
            pynini.closure(graph_integer + delete_extra_space, 0, 1)
            + graph_fractional_separator
            + delete_extra_space
            + graph_fractional
        )
        final_graph = optional_graph_negative + final_graph_wo_sign

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign, cardinal.graph_hundreds_component_at_least_one_none_zero_digit
        )
        final_graph |= optional_graph_negative + get_quantity(
            final_graph_wo_sign, cardinal.graph_hundreds_component_at_least_one_none_zero_digit
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
