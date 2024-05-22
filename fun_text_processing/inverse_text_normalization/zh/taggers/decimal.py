import pynini
from fun_text_processing.inverse_text_normalization.zh.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.zh.graph_utils import (
    DAMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


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

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_dot = pynini.string_file(get_abs_path("data/numbers/dot.tsv"))

        graph_digits = graph_digit | graph_zero

        graph_decimal = graph_digits.plus

        self.graph = graph_decimal

        point = pynini.cross("点", "")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ")
            + (pynini.cross("负", '"true"') | pynini.cross("负的", '"true"'))
            + delete_extra_space,
            0,
            1,
        )

        graph_fractional = (
            pynutil.insert('fractional_part: "') + graph_decimal + pynutil.insert('"')
        )
        graph_integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        final_graph_wo_sign = (
            pynini.closure(graph_integer, 0, 1) + point + pynutil.insert(" ") + graph_fractional
        )

        final_graph = optional_graph_negative + final_graph_wo_sign

        self.final_graph_wo_negative = final_graph_wo_sign

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
