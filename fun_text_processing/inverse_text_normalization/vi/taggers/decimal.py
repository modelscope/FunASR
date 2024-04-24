import pynini
from fun_text_processing.inverse_text_normalization.vi.graph_utils import (
    DAMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from fun_text_processing.inverse_text_normalization.vi.utils import get_abs_path
from pynini.lib import pynutil

graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))


def get_quantity(
    decimal: "pynini.FstLike", cardinal_up_to_hundred: "pynini.FstLike"
) -> "pynini.FstLike":
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. một triệu -> integer_part: "1" quantity: "triệu"
    e.g. một tỷ rưỡi -> integer_part: "1" fractional_part: "5" quantity: "tỷ"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = cardinal_up_to_hundred @ (
        pynutil.delete(pynini.closure("0"))
        + pynini.difference(DAMO_DIGIT, "0")
        + pynini.closure(DAMO_DIGIT)
    )
    suffix = pynini.union("triệu", "tỉ", "tỷ", "vạn")
    graph_four = pynini.cross("tư", "4")
    graph_one = pynini.cross("mốt", "1")
    graph_half = pynini.cross("rưỡi", "5")
    last_digit_exception = pynini.project(pynini.cross("năm", "5"), "input")
    last_digit = pynini.union(
        (pynini.project(graph_digit, "input") - last_digit_exception.arcsort()) @ graph_digit,
        graph_one,
        graph_four,
        graph_half,
    )
    optional_fraction_graph = pynini.closure(
        delete_extra_space
        + pynutil.insert('fractional_part: "')
        + (last_digit | graph_half | graph_one | graph_four)
        + pynutil.insert('"'),
        0,
        1,
    )

    res = (
        pynutil.insert('integer_part: "')
        + numbers
        + pynutil.insert('"')
        + delete_extra_space
        + pynutil.insert('quantity: "')
        + suffix
        + pynutil.insert('"')
        + optional_fraction_graph
    )
    res |= (
        decimal
        + delete_extra_space
        + pynutil.insert('quantity: "')
        + (suffix | "ngàn" | "nghìn")
        + pynutil.insert('"')
    )
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        e.g. âm hai hai phẩy không năm tư năm tỉ -> decimal { negative: "true" integer_part: "22"  fractional_part: "054" quantity: "tỉ" }
        e.g. không chấm ba lăm -> decimal { integer_part: "0" fractional_part: "35" }
        e.g. một triệu rưỡi -> decimal { integer_part: "1" quantity: "triệu" fractional_part: "5" }
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_decimal = graph_digit | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_one = pynini.cross("mốt", "1")
        graph_four = pynini.cross("tư", "4")
        graph_five = pynini.cross("lăm", "5")

        graph_decimal = pynini.union(
            graph_decimal,
            graph_four,
            pynini.closure(graph_decimal + delete_space, 1)
            + (graph_decimal | graph_four | graph_five | graph_one),
        )
        self.graph = graph_decimal

        point = pynutil.delete("chấm") | pynutil.delete("phẩy")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ")
            + pynini.cross(pynini.union("âm", "trừ"), '"true"')
            + delete_extra_space,
            0,
            1,
        )

        graph_fractional = (
            pynutil.insert('fractional_part: "') + graph_decimal + pynutil.insert('"')
        )
        graph_integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        final_graph_wo_sign = (
            pynini.closure(graph_integer + delete_extra_space, 0, 1)
            + point
            + delete_extra_space
            + graph_fractional
        )
        final_graph = optional_graph_negative + final_graph_wo_sign

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign,
            cardinal.graph_hundred_component_at_least_one_none_zero_digit,
        )
        final_graph |= optional_graph_negative + get_quantity(
            final_graph_wo_sign,
            cardinal.graph_hundred_component_at_least_one_none_zero_digit,
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
