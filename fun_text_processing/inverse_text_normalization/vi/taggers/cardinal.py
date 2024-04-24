import pynini
from fun_text_processing.inverse_text_normalization.vi.graph_utils import (
    DAMO_DIGIT,
    DAMO_SPACE,
    GraphFst,
    delete_space,
)
from fun_text_processing.inverse_text_normalization.vi.utils import get_abs_path
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. trừ hai mươi ba -> cardinal { integer: "23" negative: "-" } }
        e.g. hai nghìn lẻ chín -> cardinal { integer: "2009"} }
    Numbers below ten are not converted.
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))

        graph_one = pynini.cross("mốt", "1")
        graph_four = pynini.cross("tư", "4")
        graph_five = pynini.cross("lăm", "5")
        graph_half = pynini.cross("rưỡi", "5")
        graph_hundred = pynini.cross("trăm", "")
        graph_ten = pynini.cross("mươi", "")
        zero = pynini.cross(pynini.union("linh", "lẻ"), "0")

        optional_ten = pynini.closure(delete_space + graph_ten, 0, 1)
        last_digit_exception = pynini.project(pynini.cross("năm", "5"), "input")
        last_digit = pynini.union(
            (pynini.project(graph_digit, "input") - last_digit_exception.arcsort()) @ graph_digit,
            graph_one,
            graph_four,
            graph_five,
        )

        graph_hundred_ties_component = (graph_digit | graph_zero) + delete_space + graph_hundred
        graph_hundred_ties_component += delete_space
        graph_hundred_ties_component += pynini.union(
            graph_teen,
            (graph_half | graph_four | graph_one) + pynutil.insert("0"),
            graph_ties + optional_ten + ((delete_space + last_digit) | pynutil.insert("0")),
            zero + delete_space + (graph_digit | graph_four),
            pynutil.insert("00"),
        )
        graph_hundred_ties_component |= (
            pynutil.insert("0")
            + delete_space
            + pynini.union(
                graph_teen,
                graph_ties + optional_ten + delete_space + last_digit,
                graph_ties + delete_space + graph_ten + pynutil.insert("0"),
                zero + delete_space + (graph_digit | graph_four),
            )
        )
        graph_hundred_component = graph_hundred_ties_component | (
            pynutil.insert("00") + delete_space + graph_digit
        )

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(DAMO_DIGIT) + (DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )
        graph_hundred_ties_zero = graph_hundred_ties_component | pynutil.insert("000")

        graph_thousands = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete(pynini.union("nghìn", "ngàn")),
            pynutil.insert("000", weight=0.1),
        )

        graph_ten_thousand = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("vạn"),
            pynutil.insert("0000", weight=0.1),
        )

        graph_ten_thousand_suffix = pynini.union(
            graph_digit + delete_space + pynutil.delete(pynini.union("nghìn", "ngàn")),
            pynutil.insert("0", weight=0.1),
        )

        graph_million = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("triệu"),
            pynutil.insert("000", weight=0.1),
        )
        graph_billion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete(pynini.union("tỉ", "tỷ")),
            pynutil.insert("000", weight=0.1),
        )

        graph = pynini.union(
            graph_billion
            + delete_space
            + graph_million
            + delete_space
            + graph_thousands
            + delete_space
            + graph_hundred_ties_zero,
            graph_ten_thousand
            + delete_space
            + graph_ten_thousand_suffix
            + delete_space
            + graph_hundred_ties_zero,
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete(pynini.union("nghìn", "ngàn"))
            + delete_space
            + (((last_digit | graph_half) + pynutil.insert("00")) | graph_hundred_ties_zero),
            graph_digit,
            graph_zero,
        )

        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0"))
            + pynini.difference(DAMO_DIGIT, "0")
            + pynini.closure(DAMO_DIGIT),
            "0",
        )

        # don't convert cardinals from zero to nine inclusive
        graph_exception = pynini.project(pynini.union(graph_digit, graph_zero), "input")

        self.graph_no_exception = graph

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ")
            + pynini.cross(pynini.union("âm", "trừ"), '"-"')
            + DAMO_SPACE,
            0,
            1,
        )

        final_graph = (
            optional_minus_graph + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
