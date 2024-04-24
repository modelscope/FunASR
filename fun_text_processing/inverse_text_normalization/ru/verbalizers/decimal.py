import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    DAMO_SPACE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "true" integer_part: "3," fractional_part: "2" } -> -3,2

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.closure(pynini.cross('negative: "true" ', "-"), 0, 1)

        integer = pynutil.delete(' "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        integer_part = pynutil.delete("integer_part:") + integer
        fractional_part = pynutil.delete("fractional_part:") + integer
        optional_quantity = pynini.closure(
            pynini.accep(DAMO_SPACE) + pynutil.delete("quantity:") + integer, 0, 1
        )

        graph = optional_sign + integer_part + delete_space + fractional_part + optional_quantity
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
