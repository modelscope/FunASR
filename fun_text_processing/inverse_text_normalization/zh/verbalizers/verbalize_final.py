import pynini
from fun_text_processing.inverse_text_normalization.zh.verbalizers.verbalize import VerbalizeFst
from fun_text_processing.inverse_text_normalization.zh.verbalizers.word import WordFst
from fun_text_processing.inverse_text_normalization.zh.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


class VerbalizeFinalFst(GraphFst):
    """
    Finite state transducer that verbalizes an entire sentence, e.g.
    tokens { name: "its" } tokens { time { hours: "12" minutes: "30" } } tokens { name: "now" } -> its 12:30 now
    """

    def __init__(self):
        super().__init__(name="verbalize_final", kind="verbalize")
        verbalize = VerbalizeFst().fst
        word = WordFst().fst
        types = verbalize | word
        graph = (
            pynutil.delete("tokens")
            + delete_space
            + pynutil.delete("{")
            + delete_space
            + types
            + delete_space
            + pynutil.delete("}")
        )
        graph = delete_space + pynini.closure(graph + delete_extra_space) + graph + delete_space
        self.fst = graph
