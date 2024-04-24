import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
        e.g. fraction { integer_part: "1" numerator: "2" denominator: "3" } } -> 1 2/3

    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")
        optional_sign = pynini.closure(pynini.cross('negative: "true"', "-") + delete_space, 0, 1)
        integer = (
            pynutil.delete('integer_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + insert_space
        )
        numerator = (
            pynutil.delete('numerator: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )

        denominator = (
            pynutil.insert("/")
            + pynutil.delete('denominator: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph = (
            pynini.closure(integer + delete_space, 0, 1) + numerator + delete_space + denominator
        ).optimize()
        self.numbers = graph
        delete_tokens = self.delete_tokens(optional_sign + graph)
        self.fst = delete_tokens.optimize()
