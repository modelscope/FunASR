import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal,
        e.g. decimal { negative: "true" integer_part: "1" morphosyntactic_features: ","  fractional_part: "26" } -> -1,26
        e.g. decimal { negative: "true" integer_part: "1" morphosyntactic_features: "."  fractional_part: "26" } -> -1.26
        e.g. decimal { negative: "false" integer_part: "1" morphosyntactic_features: ","  fractional_part: "26" quantity: "millón" } -> 1,26 millón
        e.g. decimal { negative: "false" integer_part: "2" quantity: "millones" } -> 2 millones
    """

    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")
        optionl_sign = pynini.closure(pynini.cross('negative: "true"', "-") + delete_space, 0, 1)
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_integer = pynini.closure(integer + delete_space, 0, 1)

        decimal_point = pynini.cross('morphosyntactic_features: ","', ",")
        decimal_point |= pynini.cross('morphosyntactic_features: "."', ".")

        fractional = (
            decimal_point
            + delete_space
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_fractional = pynini.closure(fractional + delete_space, 0, 1)
        quantity = (
            pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_quantity = pynini.closure(pynutil.insert(" ") + quantity + delete_space, 0, 1)
        graph = optional_integer + optional_fractional + optional_quantity
        self.numbers = graph
        graph = optionl_sign + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
