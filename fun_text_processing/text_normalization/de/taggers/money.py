import pynini
from fun_text_processing.text_normalization.de.utils import get_abs_path, load_labels
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_SIGMA,
    GraphFst,
    convert_space,
    insert_space,
)
from pynini.lib import pynutil

min_singular = pynini.string_file(get_abs_path("data/money/currency_minor_singular.tsv"))
min_plural = pynini.string_file(get_abs_path("data/money/currency_minor_plural.tsv"))
maj_singular = pynini.string_file((get_abs_path("data/money/currency.tsv")))


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        "€1" -> money { currency_maj: "euro" integer_part: "ein"}
        "€1,000" -> money { currency_maj: "euro" integer_part: "ein" }
        "€1,001" -> money { currency_maj: "euro" integer_part: "eins" fractional_part: "null null eins"}
        "£1,4" -> money { integer_part: "ein" currency_maj: "pfund" fractional_part: "vierzig" preserve_order: true}
               -> money { integer_part: "ein" currency_maj: "pfund" fractional_part: "vierzig" currency_min: "pence" preserve_order: true}
        "£0,01" -> money { fractional_part: "ein" currency_min: "penny" preserve_order: true}
        "£0,01 million" -> money { currency_maj: "pfund" integer_part: "null" fractional_part: "null eins" quantity: "million"}

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph
        graph_decimal_final = decimal.fst

        maj_singular_labels = load_labels(get_abs_path("data/money/currency.tsv"))
        maj_singular_graph = convert_space(maj_singular)
        maj_plural_graph = maj_singular_graph

        graph_maj_singular = (
            pynutil.insert('currency_maj: "') + maj_singular_graph + pynutil.insert('"')
        )
        graph_maj_plural = (
            pynutil.insert('currency_maj: "') + maj_plural_graph + pynutil.insert('"')
        )

        optional_delete_fractional_zeros = pynini.closure(
            pynutil.delete(",") + pynini.closure(pynutil.delete("0"), 1), 0, 1
        )
        graph_integer_one = (
            pynutil.insert('integer_part: "') + pynini.cross("1", "ein") + pynutil.insert('"')
        )

        # only for decimals where third decimal after comma is non-zero or with quantity
        decimal_delete_last_zeros = (
            pynini.closure(DAMO_DIGIT, 1)
            + pynini.accep(",")
            + pynini.closure(DAMO_DIGIT, 2)
            + (DAMO_DIGIT - "0")
            + pynini.closure(pynutil.delete("0"))
        )
        decimal_with_quantity = DAMO_SIGMA + DAMO_ALPHA
        graph_decimal = (
            graph_maj_plural
            + insert_space
            + (decimal_delete_last_zeros | decimal_with_quantity) @ graph_decimal_final
        )

        graph_integer = (
            pynutil.insert('integer_part: "')
            + ((DAMO_SIGMA - "1") @ cardinal_graph)
            + pynutil.insert('"')
        )

        graph_integer_only = graph_maj_singular + insert_space + graph_integer_one
        graph_integer_only |= graph_maj_plural + insert_space + graph_integer

        graph = (graph_integer_only + optional_delete_fractional_zeros) | graph_decimal

        # remove trailing zeros of non zero number in the first 2 digits and fill up to 2 digits
        # e.g. 2000 -> 20, 0200->02, 01 -> 01, 10 -> 10
        # not accepted: 002, 00, 0,
        two_digits_fractional_part = (
            pynini.closure(DAMO_DIGIT) + (DAMO_DIGIT - "0") + pynini.closure(pynutil.delete("0"))
        ) @ (
            (pynutil.delete("0") + (DAMO_DIGIT - "0"))
            | ((DAMO_DIGIT - "0") + pynutil.insert("0"))
            | ((DAMO_DIGIT - "0") + DAMO_DIGIT)
        )

        graph_min_singular = pynutil.insert(' currency_min: "') + min_singular + pynutil.insert('"')
        graph_min_plural = pynutil.insert(' currency_min: "') + min_plural + pynutil.insert('"')

        # format ** euro ** cent
        decimal_graph_with_minor = None
        for curr_symbol, _ in maj_singular_labels:
            preserve_order = pynutil.insert(" preserve_order: true")
            integer_plus_maj = (
                graph_integer + insert_space + pynutil.insert(curr_symbol) @ graph_maj_plural
            )
            integer_plus_maj |= (
                graph_integer_one + insert_space + pynutil.insert(curr_symbol) @ graph_maj_singular
            )
            # non zero integer part
            integer_plus_maj = (pynini.closure(DAMO_DIGIT) - "0") @ integer_plus_maj

            graph_fractional_one = two_digits_fractional_part @ pynini.cross("1", "ein")
            graph_fractional_one = (
                pynutil.insert('fractional_part: "') + graph_fractional_one + pynutil.insert('"')
            )
            graph_fractional = (
                two_digits_fractional_part
                @ (pynini.closure(DAMO_DIGIT, 1, 2) - "1")
                @ cardinal.two_digit_non_zero
            )
            graph_fractional = (
                pynutil.insert('fractional_part: "') + graph_fractional + pynutil.insert('"')
            )

            fractional_plus_min = (
                graph_fractional + insert_space + pynutil.insert(curr_symbol) @ graph_min_plural
            )
            fractional_plus_min |= (
                graph_fractional_one
                + insert_space
                + pynutil.insert(curr_symbol) @ graph_min_singular
            )

            decimal_graph_with_minor_curr = (
                integer_plus_maj + pynini.cross(",", " ") + fractional_plus_min
            )
            decimal_graph_with_minor_curr |= pynutil.add_weight(
                integer_plus_maj
                + pynini.cross(",", " ")
                + pynutil.insert('fractional_part: "')
                + two_digits_fractional_part @ cardinal.two_digit_non_zero
                + pynutil.insert('"'),
                weight=0.0001,
            )

            decimal_graph_with_minor_curr |= pynutil.delete("0,") + fractional_plus_min
            decimal_graph_with_minor_curr = (
                pynutil.delete(curr_symbol) + decimal_graph_with_minor_curr + preserve_order
            )

            decimal_graph_with_minor = (
                decimal_graph_with_minor_curr
                if decimal_graph_with_minor is None
                else pynini.union(decimal_graph_with_minor, decimal_graph_with_minor_curr)
            )

        final_graph = graph | decimal_graph_with_minor

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
