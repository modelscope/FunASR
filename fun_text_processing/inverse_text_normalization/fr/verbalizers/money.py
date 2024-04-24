import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "12" fractional_part: "05" currency: "$" } -> 12.05 $

    Args:
        decimal: DecimalFst
    """

    def __init__(self, decimal: GraphFst):
        super().__init__(name="money", kind="verbalize")
        unit = (
            pynutil.delete("currency:")
            + delete_extra_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        graph = decimal.numbers + delete_space + unit
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
