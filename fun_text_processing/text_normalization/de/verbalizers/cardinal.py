import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinals
        e.g. cardinal { integer: "zwei" } -> "zwei"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)
        optional_sign = pynini.closure(pynini.cross('negative: "true" ', "minus "), 0, 1)
        self.optional_sign = optional_sign
        integer = pynini.closure(DAMO_NOT_QUOTE, 1)

        self.integer = pynutil.delete(' "') + integer + pynutil.delete('"')

        integer = pynutil.delete("integer:") + self.integer
        self.numbers = integer
        graph = optional_sign + self.numbers
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
