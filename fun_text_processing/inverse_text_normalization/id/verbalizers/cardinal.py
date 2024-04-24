import pynini
from fun_text_processing.inverse_text_normalization.id.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal
        e.g. cardinal { integer: "23" negative: "-" } -> -23
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="verbalize")
        optional_sign = pynini.closure(
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete('"')
            + DAMO_NOT_QUOTE
            + pynutil.delete('"')
            + delete_space,
            0,
            1,
        )
        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        self.numbers = graph
        graph = optional_sign + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
