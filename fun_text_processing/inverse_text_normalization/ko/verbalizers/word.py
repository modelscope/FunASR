import pynini
from fun_text_processing.inverse_text_normalization.ko.graph_utils import (
    DAMO_CHAR,
    DAMO_SIGMA,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class WordFst(GraphFst):
    """
    Finite state transducer for verbalizing plain tokens
        e.g. tokens { name: "sleep" } -> sleep
    """

    def __init__(self):
        super().__init__(name="word", kind="verbalize")
        chars = pynini.closure(DAMO_CHAR - " ", 1)
        char = (
            pynutil.delete("name:")
            + delete_space
            + pynutil.delete('"')
            + chars
            + pynutil.delete('"')
        )
        graph = char @ pynini.cdrewrite(pynini.cross("\u00A0", " "), "", "", DAMO_SIGMA)

        self.fst = graph.optimize()
