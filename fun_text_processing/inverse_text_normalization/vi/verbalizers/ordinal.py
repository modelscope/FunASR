import pynini
from fun_text_processing.inverse_text_normalization.vi.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing ordinal, e.g.
       ordinal { integer: "2" } -> thứ 2
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")
        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph = pynutil.insert("thứ ") + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
