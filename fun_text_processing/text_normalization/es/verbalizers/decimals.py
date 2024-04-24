import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.es import LOCALIZATION
from fun_text_processing.text_normalization.es.graph_utils import (
    add_cardinal_apocope_fem,
    shift_cardinal_gender,
    shift_number_gender,
    strip_cardinal_apocope,
)
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
        Finite state transducer for classifying decimal, e.g.
                decimal { negative: "true" integer_part: "dos"  fractional_part: "cuatro cero" quantity: "billones" } -> menos dos coma quatro cero billones
                decimal { integer_part: "un" quantity: "billón" } -> un billón

    Args:
                deterministic: if True will provide a single transduction option,
                        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        optional_sign = pynini.closure(
            pynini.cross('negative: "true"', "menos ") + delete_space, 0, 1
        )
        integer = (
            pynutil.delete('integer_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        fractional_default = (
            pynutil.delete('fractional_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        conjunction = (
            pynutil.insert(" punto ") if LOCALIZATION == "am" else pynutil.insert(" coma ")
        )
        if not deterministic:
            conjunction |= pynutil.insert(pynini.union(" con ", " y "))
            fractional_default |= strip_cardinal_apocope(fractional_default)
        fractional = conjunction + fractional_default

        quantity = (
            delete_space
            + insert_space
            + pynutil.delete('quantity: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_quantity = pynini.closure(quantity, 0, 1)

        graph_masc = optional_sign + pynini.union(
            (integer + quantity), (integer + delete_space + fractional + optional_quantity)
        )

        # Allowing permutation for fem gender, don't include quantity since "million","billion", etc.. are masculine
        graph_fem = optional_sign + (
            shift_cardinal_gender(integer) + delete_space + shift_number_gender(fractional)
        )
        if not deterministic:  # "una" will drop to "un" in certain cases
            graph_fem |= add_cardinal_apocope_fem(graph_fem)

        self.numbers_only_quantity = (
            optional_sign
            + pynini.union(
                (integer + quantity), (integer + delete_space + fractional + quantity)
            ).optimize()
        )

        self.graph_masc = (graph_masc + delete_preserve_order).optimize()
        self.graph_fem = (graph_fem + delete_preserve_order).optimize()

        graph = graph_masc | graph_fem

        graph += delete_preserve_order
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
