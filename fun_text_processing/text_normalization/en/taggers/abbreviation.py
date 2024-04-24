import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_UPPER, GraphFst, insert_space
from pynini.lib import pynutil


class AbbreviationFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. "ABC" -> tokens { abbreviation { value: "A B C" } }

    Args:
        whitelist: whitelist FST
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, whitelist: "pynini.FstLike", deterministic: bool = True):
        super().__init__(name="abbreviation", kind="classify", deterministic=deterministic)

        dot = pynini.accep(".")
        # A.B.C. -> A. B. C.
        graph = DAMO_UPPER + dot + pynini.closure(insert_space + DAMO_UPPER + dot, 1)
        # A.B.C. -> A.B.C.
        graph |= DAMO_UPPER + dot + pynini.closure(DAMO_UPPER + dot, 1)
        # ABC -> A B C
        graph |= DAMO_UPPER + pynini.closure(insert_space + DAMO_UPPER, 1)

        # exclude words that are included in the whitelist
        graph = pynini.compose(
            pynini.difference(
                pynini.project(graph, "input"), pynini.project(whitelist.graph, "input")
            ),
            graph,
        )

        graph = pynutil.insert('value: "') + graph.optimize() + pynutil.insert('"')
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
