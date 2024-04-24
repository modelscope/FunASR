import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "true" integer_part: "twelve" fractional_part: "five o o six" quantity: "billion" } -> minus twelve point five o o six billion

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)
        self.optional_sign = pynini.cross('negative: "true"', "minus ")
        if not deterministic:
            self.optional_sign |= pynini.cross('negative: "true"', "negative ")
        self.optional_sign = pynini.closure(self.optional_sign + delete_space, 0, 1)
        self.integer = pynutil.delete("integer_part:") + cardinal.integer
        self.optional_integer = pynini.closure(self.integer + delete_space + insert_space, 0, 1)
        self.fractional_default = (
            pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        self.fractional = pynutil.insert("point ") + self.fractional_default

        self.quantity = (
            delete_space
            + insert_space
            + pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        self.optional_quantity = pynini.closure(self.quantity, 0, 1)

        graph = self.optional_sign + (
            self.integer
            | (self.integer + self.quantity)
            | (self.optional_integer + self.fractional + self.optional_quantity)
        )

        self.numbers = graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
