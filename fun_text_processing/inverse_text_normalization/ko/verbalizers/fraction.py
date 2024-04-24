from fun_text_processing.inverse_text_normalization.ko.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)

import pynini
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction,
    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")
        optional_sign = pynini.closure(pynini.cross('negative: "true"', "-") + delete_space, 0, 1)
        numerator = (
            pynutil.delete("numerator:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        denominator = (
            pynutil.delete("denominator:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph = (
            optional_sign + numerator + delete_space + pynutil.insert("/") + denominator
        ).optimize()
        self.numbers = graph
        delete_tokens = self.delete_tokens(optional_sign + graph)
        self.fst = delete_tokens.optimize()
