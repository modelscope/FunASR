import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class AbbreviationFst(GraphFst):
    """
    Finite state transducer for verbalizing abbreviations
        e.g. tokens { abbreviation { value: "A B C" } } -> "ABC"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="abbreviation", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete('value: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
