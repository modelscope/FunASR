import pynini
from fun_text_processing.inverse_text_normalization.zh.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.zh.graph_utils import (
    DAMO_DIGIT,
    DAMO_NOT_SPACE,
    DAMO_SIGMA,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    get_singulars,
    insert_space,
)
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money
        e.g. twelve dollars and five cents -> money { integer_part: "12" fractional_part: 05 currency: "$" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")
        # quantity, integer_part, fractional_part, currency

        cardinal_graph = cardinal.graph_no_exception
        decimal_graph = decimal.final_graph_wo_negative
        graph_code = pynini.string_file(get_abs_path("data/money/code.tsv"))
        graph_symbol = pynini.string_file(get_abs_path("data/money/symbol.tsv"))

        graph_unit = (
            pynutil.insert('currency: "') + (graph_code | graph_symbol) + pynutil.insert('"')
        )

        graph_integer = (
            pynutil.insert('integer_part: "')
            + cardinal_graph
            + pynutil.insert('"')
            + pynutil.insert(" ")
            + graph_unit
        )

        graph_decimal = decimal_graph + pynutil.insert(" ") + graph_unit

        final_graph = graph_integer | graph_decimal

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
