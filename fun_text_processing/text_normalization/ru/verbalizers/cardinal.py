import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinals
        e.g. cardinal { integer: "тысяча один" } -> "тысяча один"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)
        optional_sign = pynini.closure(pynini.cross('negative: "true" ', "минус "), 0, 1)
        optional_quantity_part = pynini.closure(
            pynini.accep(" ")
            + pynutil.delete('quantity: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"'),
            0,
            1,
        )
        integer = (
            pynutil.delete('integer: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )
        self.graph = optional_sign + integer + optional_quantity_part
        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
