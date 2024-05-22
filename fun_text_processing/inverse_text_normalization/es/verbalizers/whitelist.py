import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_CHAR,
    DAMO_SIGMA,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class WhiteListFst(GraphFst):
    """
    Finite state transducer for verbalizing whitelist
        e.g. tokens { name: "uds." } -> uds.
    """

    def __init__(self):
        super().__init__(name="whitelist", kind="verbalize")
        graph = (
            pynutil.delete("name:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_CHAR - " ", 1)
            + pynutil.delete('"')
        )
        graph = graph @ pynini.cdrewrite(pynini.cross("\u00A0", " "), "", "", DAMO_SIGMA)
        self.fst = graph.optimize()
