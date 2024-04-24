import pynini
from fun_text_processing.text_normalization.de.taggers.decimal import quantities
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    insert_space,
)
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        decimal { negative: "true" integer_part: "elf"  fractional_part: "vier null sechs" quantity: "billionen" } -> minus elf komma vier null sechs billionen
        decimal { integer_part: "eins" quantity: "billion" } -> eins billion

    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        delete_space = pynutil.delete(" ")
        self.optional_sign = pynini.closure(
            pynini.cross('negative: "true"', "minus ") + delete_space, 0, 1
        )
        self.integer = (
            pynutil.delete('integer_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        self.fractional_default = (
            pynutil.delete('fractional_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        self.fractional = pynutil.insert(" komma ") + self.fractional_default

        self.quantity = (
            delete_space
            + insert_space
            + pynutil.delete('quantity: "')
            + quantities
            + pynutil.delete('"')
        )
        self.optional_quantity = pynini.closure(self.quantity, 0, 1)

        graph = self.optional_sign + (
            self.integer + self.quantity
            | self.integer + delete_space + self.fractional + self.optional_quantity
        )

        self.numbers = graph
        graph += delete_preserve_order
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
