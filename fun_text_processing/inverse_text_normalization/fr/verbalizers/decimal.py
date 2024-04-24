import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    DAMO_DIGIT,
    DAMO_NON_BREAKING_SPACE,
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class NumberParser(GraphFst):
    """
    Finite state transducer for parsing strings of digis. Breaks up digit strings into groups of three for
        strings of digits of four or more (inclusive). Groupings are separated by non-breaking space.
    e.g. '1000' -> '1 000'
    e.g. '1000,33333' -> '1 000,333 33
    """

    def __init__(self):
        super().__init__(name="parser", kind="verbalize")


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "true" integer_part: "12"  fractional_part: "5006" quantity: "billion" } -> -12.5006 billion
    """

    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")

        # Need parser to group digits by threes
        exactly_three_digits = DAMO_DIGIT**3
        at_most_three_digits = pynini.closure(DAMO_DIGIT, 1, 3)

        space_every_three_integer = (
            at_most_three_digits
            + (pynutil.insert(DAMO_NON_BREAKING_SPACE) + exactly_three_digits).closure()
        )
        space_every_three_decimal = (
            pynini.accep(",")
            + (exactly_three_digits + pynutil.insert(DAMO_NON_BREAKING_SPACE)).closure()
            + at_most_three_digits
        )
        group_by_threes = space_every_three_integer | space_every_three_decimal
        self.group_by_threes = group_by_threes

        optional_sign = pynini.closure(pynini.cross('negative: "true"', "-") + delete_space, 0, 1)
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        integer = integer @ group_by_threes
        optional_integer = pynini.closure(integer + delete_space, 0, 1)
        fractional = (
            pynutil.insert(",")
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        fractional = fractional @ group_by_threes
        optional_fractional = pynini.closure(fractional + delete_space, 0, 1)
        quantity = (
            pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_quantity = pynini.closure(pynutil.insert(" ") + quantity + delete_space, 0, 1)
        graph = (optional_integer + optional_fractional + optional_quantity).optimize()
        self.numbers = graph
        graph = optional_sign + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
