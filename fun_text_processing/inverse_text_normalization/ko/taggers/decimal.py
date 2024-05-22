import pynini
from fun_text_processing.inverse_text_normalization.ko.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.ko.graph_utils import (
    DAMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


def get_quantity(
    decimal: "pynini.FstLike", cardinal_up_to_hundred: "pynini.FstLike"
) -> "pynini.FstLike":
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. one million -> integer_part: "1" quantity: "million"
    e.g. one point five million -> integer_part: "1" fractional_part: "5" quantity: "million"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = cardinal_up_to_hundred @ (
        pynutil.delete(pynini.closure("0"))
        + pynini.difference(DAMO_DIGIT, "0")
        + pynini.closure(DAMO_DIGIT)
    )
    # "만", "백만", "천만", "억", "조", 万、百万、千万、亿、兆
    # 천 千
    suffix = pynini.union("만", "백만", "천만", "억", "조")
    res = (
        pynutil.insert('integer_part: "')
        + numbers
        + pynutil.insert('"')
        + delete_extra_space
        + pynutil.insert('quantity: "')
        + suffix
        + pynutil.insert('"')
    )
    res |= (
        decimal
        + delete_extra_space
        + pynutil.insert('quantity: "')
        + (suffix | "천")
        + pynutil.insert('"')
    )
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        e.g. minus twelve point five o o six billion -> decimal { negative: "true" integer_part: "12"  fractional_part: "5006" quantity: "billion" }
        e.g. one billion -> decimal { integer_part: "1" quantity: "billion" }
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_decimal = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_decimal |= pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        graph_decimal = pynini.closure(graph_decimal)
        self.graph = graph_decimal

        ##마이너스 负
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("마이너스", '"true"') + delete_extra_space,
            0,
            1,
        )

        graph_fractional = (
            pynutil.insert('fractional_part: "') + graph_decimal + pynutil.insert('"')
        )

        # 점 点
        graph_integer = (
            pynutil.insert('integer_part: "')
            + cardinal_graph
            + pynutil.delete("점")
            + pynutil.insert('"')
        )

        final_graph_wo_sign = graph_integer + pynini.cross(" ", " ") + graph_fractional

        final_graph = optional_graph_negative + delete_space + final_graph_wo_sign

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign, cardinal.graph_hundred_component_at_least_one_none_zero_digit
        )
        final_graph |= optional_graph_negative + get_quantity(
            final_graph_wo_sign, cardinal.graph_hundred_component_at_least_one_none_zero_digit
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
