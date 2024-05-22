import pynini
from fun_text_processing.inverse_text_normalization.ja.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    DAMO_SIGMA,
    delete_extra_space,
)
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction,
        e.g. fraction { numerator: "2" denominator: "3" } } -> 2/3
        e.g. fraction { numerator: "20" denominator: "3" negative: "true"} } -> 2/3
    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")
        optional_sign = pynini.closure(pynini.cross('negative: "true"', "-") + delete_space, 0, 1)
        numerator = (
            pynutil.delete('numerator: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )

        denominator = (
            pynutil.insert("/")
            + pynutil.delete('denominator: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph = (numerator + delete_space + denominator).optimize()

        # numerator = pynutil.delete('numerator: "') + DAMO_NOT_QUOTE + pynutil.delete('"')
        #
        # denominator = (
        #     pynutil.delete('denominator: "')
        #     + DAMO_NOT_QUOTE
        #     + pynutil.delete('"')
        # )
        #
        # graph = (numerator + pynutil.insert("/") + denominator).optimize()

        self.numbers = graph
        delete_tokens = self.delete_tokens(optional_sign + graph)
        self.fst = delete_tokens.optimize()
