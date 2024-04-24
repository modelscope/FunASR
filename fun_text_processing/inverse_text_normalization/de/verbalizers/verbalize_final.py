import pynini
from fun_text_processing.inverse_text_normalization.de.verbalizers.verbalize import VerbalizeFst
from fun_text_processing.inverse_text_normalization.en.verbalizers.word import WordFst
from fun_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


class VerbalizeFinalFst(GraphFst):
    """
    Finite state transducer that verbalizes an entire sentence, e.g.
    tokens { name: "jetzt" } tokens { name: "ist" } tokens { time { hours: "12" minutes: "30" } } -> jetzt ist 12:30 Uhr
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize_final", kind="verbalize", deterministic=deterministic)
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
