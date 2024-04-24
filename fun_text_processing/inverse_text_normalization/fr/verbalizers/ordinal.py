import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    DAMO_DIGIT,
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from fun_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing ordinal, e.g.
        ordinal { integer: "13" morphosyntactic_features: "e" } -> 13ᵉ

    Given 'special' terms for ordinals (e.g. siècle), renders
        amount in conventional format. e.g.

        ordinal { integer: "13" morphosyntactic_features: "e/siècle" } -> XIIIᵉ
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")
        graph_integer = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        replace_suffix = pynini.union(
            pynini.cross("e", "ᵉ"),  # only delete first quote since there may be more features
            pynini.cross("d", "ᵈ"),
            pynini.cross("r", "ʳ"),
            pynini.cross("s", "ˢ"),
        )
        replace_suffix = pynutil.delete(' morphosyntactic_features: "') + replace_suffix.plus

        graph_arabic = graph_integer + replace_suffix.plus

        # For roman.
        graph_roman_digits = pynini.string_file(
            get_abs_path("data/roman/digits_large.tsv")
        ).invert()
        graph_roman_ties = pynini.string_file(get_abs_path("data/roman/ties_large.tsv")).invert()
        graph_roman_hundreds = pynini.string_file(
            get_abs_path("data/roman/hundreds_large.tsv")
        ).invert()
        graph_roman_zero_digit = pynutil.delete("0")

        graph_roman_hundreds = DAMO_DIGIT**3 @ (
            graph_roman_hundreds
            + pynini.union(graph_roman_ties, graph_roman_zero_digit)
            + pynini.union(graph_roman_digits, graph_roman_zero_digit)
        )
        graph_roman_ties = DAMO_DIGIT**2 @ (
            graph_roman_ties + pynini.union(graph_roman_digits, graph_roman_zero_digit)
        )
        graph_roman_digits = DAMO_DIGIT @ graph_roman_digits

        graph_roman_integers = graph_roman_hundreds | graph_roman_ties | graph_roman_digits

        graph_roman = (graph_integer @ graph_roman_integers) + replace_suffix
        graph_roman += pynini.cross("/", " ") + "siècle"

        graph = (graph_roman | graph_arabic) + pynutil.delete('"')

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
