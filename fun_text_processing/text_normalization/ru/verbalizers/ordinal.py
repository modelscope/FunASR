import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing roman numerals
        e.g. ordinal { integer: "второе" } } -> "второе"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="ordinal", kind="verbalize", deterministic=deterministic)

        value = pynini.closure(DAMO_NOT_QUOTE)
        graph = pynutil.delete('integer: "') + value + pynutil.delete('"')
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
