import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    delete_preserve_order,
)
from fun_text_processing.text_normalization.es.graph_utils import (
    shift_cardinal_gender,
    strip_cardinal_apocope,
)
from fun_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

fem = pynini.string_file((get_abs_path("data/money/currency_plural_fem.tsv")))
masc = pynini.string_file((get_abs_path("data/money/currency_plural_masc.tsv")))

fem_singular = pynini.project(fem, "input")
masc_singular = pynini.project(masc, "input")

fem_plural = pynini.project(fem, "output")
masc_plural = pynini.project(masc, "output")


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { currency_maj: "euro" integer_part: "un"} -> "un euro"
        money { currency_maj: "euro" integer_part: "un" fractional_part: "cero cero un"} -> "uno coma cero cero uno euros"
        money { integer_part: "un" currency_maj: "libra" fractional_part: "cuarenta" preserve_order: true} -> "una libra cuarenta"
        money { integer_part: "un" currency_maj: "libra" fractional_part: "cuarenta" currency_min: "peniques" preserve_order: true} -> "una libra con cuarenta peniques"
        money { fractional_part: "un" currency_min: "penique" preserve_order: true} -> "un penique"

    Args:
        decimal: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        maj_singular_masc = (
            pynutil.delete('currency_maj: "')
            + (pynini.closure(DAMO_NOT_QUOTE, 1) @ masc_singular)
            + pynutil.delete('"')
        )
        maj_singular_fem = (
            pynutil.delete('currency_maj: "')
            + (pynini.closure(DAMO_NOT_QUOTE, 1) @ fem_singular)
            + pynutil.delete('"')
        )

        maj_plural_masc = (
            pynutil.delete('currency_maj: "')
            + (pynini.closure(DAMO_NOT_QUOTE, 1) @ masc_plural)
            + pynutil.delete('"')
        )
        maj_plural_fem = (
            pynutil.delete('currency_maj: "')
            + (pynini.closure(DAMO_NOT_QUOTE, 1) @ fem_plural)
            + pynutil.delete('"')
        )

        maj_masc = maj_plural_masc | maj_singular_masc  # Tagger kept quantity resolution stable
        maj_fem = maj_plural_fem | maj_singular_fem

        min_singular_masc = (
            pynutil.delete('currency_min: "')
            + (pynini.closure(DAMO_NOT_QUOTE, 1) @ masc_singular)
            + pynutil.delete('"')
        )
        min_singular_fem = (
            pynutil.delete('currency_min: "')
            + (pynini.closure(DAMO_NOT_QUOTE, 1) @ fem_singular)
            + pynutil.delete('"')
        )

        min_plural_masc = (
            pynutil.delete('currency_min: "')
            + (pynini.closure(DAMO_NOT_QUOTE, 1) @ masc_plural)
            + pynutil.delete('"')
        )
        min_plural_fem = (
            pynutil.delete('currency_min: "')
            + (pynini.closure(DAMO_NOT_QUOTE, 1) @ fem_plural)
            + pynutil.delete('"')
        )

        min_masc = min_plural_masc | min_singular_masc
        min_fem = min_plural_fem | min_singular_fem

        fractional_part = (
            pynutil.delete('fractional_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        integer_part = (
            pynutil.delete('integer_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_add_and = pynini.closure(pynutil.insert(pynini.union("con ", "y ")), 0, 1)

        #  *** currency_maj
        graph_integer_masc = integer_part + DAMO_SPACE + maj_masc
        graph_integer_fem = shift_cardinal_gender(integer_part) + DAMO_SPACE + maj_fem

        graph_integer = graph_integer_fem | graph_integer_masc

        #  *** currency_maj + (***) | ((con) *** current_min)
        graph_integer_with_minor_masc = (
            graph_integer_masc
            + DAMO_SPACE
            + pynini.union(
                optional_add_and + strip_cardinal_apocope(fractional_part),
                (optional_add_and + fractional_part + DAMO_SPACE + min_masc),
                (optional_add_and + shift_cardinal_gender(fractional_part) + DAMO_SPACE + min_fem),
            )  # Could be minor currency that is different gender
            + delete_preserve_order
        )

        graph_integer_with_minor_fem = (
            graph_integer_fem
            + DAMO_SPACE
            + pynini.union(
                optional_add_and + shift_cardinal_gender(fractional_part),
                (optional_add_and + fractional_part + DAMO_SPACE + min_masc),
                (optional_add_and + shift_cardinal_gender(fractional_part) + DAMO_SPACE + min_fem),
            )  # Could be minor currency that is different gender
            + delete_preserve_order
        )

        graph_integer_with_minor = graph_integer_with_minor_fem | graph_integer_with_minor_masc

        ## *** coma *** currency_maj
        graph_decimal_masc = decimal.graph_masc + DAMO_SPACE + maj_masc

        graph_decimal_fem = decimal.graph_fem
        graph_decimal_fem |= (
            decimal.numbers_only_quantity
        )  # can still have "x billions" with fem currency
        graph_decimal_fem += DAMO_SPACE + maj_fem

        graph_decimal = graph_decimal_fem | graph_decimal_masc
        graph_decimal = (
            pynini.cdrewrite(
                pynutil.insert(" de"),
                'quantity: "' + pynini.closure(DAMO_NOT_QUOTE, 1),
                '"',
                DAMO_SIGMA,
            )
            @ graph_decimal
        )  # formally it's millones/billones de ***

        # *** current_min
        graph_minor_masc = fractional_part + DAMO_SPACE + min_masc + delete_preserve_order
        graph_minor_fem = (
            shift_cardinal_gender(fractional_part) + DAMO_SPACE + min_fem + delete_preserve_order
        )

        graph_minor = graph_minor_fem | graph_minor_masc

        graph = graph_integer | graph_integer_with_minor | graph_decimal | graph_minor

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
