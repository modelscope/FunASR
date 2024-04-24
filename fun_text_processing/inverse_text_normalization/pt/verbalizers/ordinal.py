import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing ordinal, e.g.
        ordinal { integer: "13" morphosyntactic_features: "o" } -> 13º
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

        replace_suffix = pynini.union(
            pynini.cross(' morphosyntactic_features: "o"', "º"),
            pynini.cross(' morphosyntactic_features: "a"', "ª"),
        )

        graph = graph + replace_suffix

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
