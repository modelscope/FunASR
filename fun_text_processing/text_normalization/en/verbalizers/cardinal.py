import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal, e.g.
        cardinal { negative: "true" integer: "23" } -> minus twenty three

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)

        self.optional_sign = pynini.cross('negative: "true"', "minus ")
        if not deterministic:
            self.optional_sign |= pynini.cross('negative: "true"', "negative ")
        self.optional_sign = pynini.closure(self.optional_sign + delete_space, 0, 1)

        integer = pynini.closure(DAMO_NOT_QUOTE)

        self.integer = delete_space + pynutil.delete('"') + integer + pynutil.delete('"')
        integer = pynutil.delete("integer:") + self.integer

        self.numbers = self.optional_sign + integer
        delete_tokens = self.delete_tokens(self.numbers)
        self.fst = delete_tokens.optimize()
