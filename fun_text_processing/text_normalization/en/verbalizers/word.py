import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_CHAR,
    DAMO_SIGMA,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class WordFst(GraphFst):
    """
    Finite state transducer for verbalizing word
        e.g. tokens { name: "sleep" } -> sleep

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="word", kind="verbalize", deterministic=deterministic)
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
