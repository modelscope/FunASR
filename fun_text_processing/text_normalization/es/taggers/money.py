import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.es.graph_utils import decimal_separator
from fun_text_processing.text_normalization.es.utils import get_abs_path, load_labels
from pynini.lib import pynutil

maj_singular_labels = load_labels(get_abs_path("data/money/currency_major.tsv"))
maj_singular = pynini.string_file((get_abs_path("data/money/currency_major.tsv")))
min_singular = pynini.string_file(get_abs_path("data/money/currency_minor.tsv"))
fem_plural = pynini.string_file((get_abs_path("data/money/currency_plural_fem.tsv")))
masc_plural = pynini.string_file((get_abs_path("data/money/currency_plural_masc.tsv")))


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        "€1" -> money { currency_maj: "euro" integer_part: "un"}
        "€1,000" -> money { currency_maj: "euro" integer_part: "un" }
        "€1,001" -> money { currency_maj: "euro" integer_part: "un" fractional_part: "cero cero un" }
        "£1,4" -> money { integer_part: "una" currency_maj: "libra" fractional_part: "cuarenta" preserve_order: true }
               -> money { integer_part: "una" currency_maj: "libra" fractional_part: "cuarenta" currency_min: "penique" preserve_order: true }
        "0,01 £" -> money { fractional_part: "un" currency_min: "penique" preserve_order: true }
        "0,02 £" -> money { fractional_part: "dos" currency_min: "peniques" preserve_order: true }
        "£0,01 million" -> money { currency_maj: "libra" integer_part: "cero" fractional_part: "cero un" quantity: "million" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph
        graph_decimal_final = decimal.final_graph_wo_negative

        maj_singular_graph = maj_singular
        min_singular_graph = min_singular
        maj_plural_graph = maj_singular @ (fem_plural | masc_plural)
        min_plural_graph = min_singular @ (fem_plural | masc_plural)

        graph_maj_singular = (
            pynutil.insert('currency_maj: "') + maj_singular_graph + pynutil.insert('"')
        )
        graph_maj_plural = (
            pynutil.insert('currency_maj: "') + maj_plural_graph + pynutil.insert('"')
        )

        graph_integer_one = (
            pynutil.insert('integer_part: "') + pynini.cross("1", "un") + pynutil.insert('"')
        )

        decimal_with_quantity = (DAMO_SIGMA + DAMO_ALPHA) @ graph_decimal_final

        graph_decimal_plural = pynini.union(
            graph_maj_plural
            + pynini.closure(delete_space, 0, 1)
            + insert_space
            + graph_decimal_final,  # $1,05
            graph_decimal_final
            + pynini.closure(delete_space, 0, 1)
            + insert_space
            + graph_maj_plural,  # 1,05 $
        )
        graph_decimal_plural = (
            (DAMO_SIGMA - "1") + decimal_separator + DAMO_SIGMA
        ) @ graph_decimal_plural  # Can't have "un euros"

        graph_decimal_singular = pynini.union(
            graph_maj_singular
            + pynini.closure(delete_space, 0, 1)
            + insert_space
            + graph_decimal_final,  # $1,05
            graph_decimal_final
            + pynini.closure(delete_space, 0, 1)
            + insert_space
            + graph_maj_singular,  # 1,05 $
        )
        graph_decimal_singular = (
            pynini.accep("1") + decimal_separator + DAMO_SIGMA
        ) @ graph_decimal_singular

        graph_decimal = pynini.union(
            graph_decimal_singular,
            graph_decimal_plural,
            graph_maj_plural
            + pynini.closure(delete_space, 0, 1)
            + insert_space
            + decimal_with_quantity,
        )

        graph_integer = (
            pynutil.insert('integer_part: "')
            + ((DAMO_SIGMA - "1") @ cardinal_graph)
            + pynutil.insert('"')
        )

        graph_integer_only = pynini.union(
            graph_maj_singular
            + pynini.closure(delete_space, 0, 1)
            + insert_space
            + graph_integer_one,
            graph_integer_one
            + pynini.closure(delete_space, 0, 1)
            + insert_space
            + graph_maj_singular,
        )
        graph_integer_only |= pynini.union(
            graph_maj_plural + pynini.closure(delete_space, 0, 1) + insert_space + graph_integer,
            graph_integer + pynini.closure(delete_space, 0, 1) + insert_space + graph_maj_plural,
        )

        graph = graph_integer_only | graph_decimal

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

        graph_min_singular = (
            pynutil.insert('currency_min: "') + min_singular_graph + pynutil.insert('"')
        )
        graph_min_plural = (
            pynutil.insert('currency_min: "') + min_plural_graph + pynutil.insert('"')
        )

        # format ** euro ** cent
        decimal_graph_with_minor = None
        for curr_symbol, _ in maj_singular_labels:
            preserve_order = pynutil.insert(" preserve_order: true")

            integer_plus_maj = pynini.union(
                graph_integer + insert_space + pynutil.insert(curr_symbol) @ graph_maj_plural,
                graph_integer_one + insert_space + pynutil.insert(curr_symbol) @ graph_maj_singular,
            )
            # non zero integer part
            integer_plus_maj = (pynini.closure(DAMO_DIGIT) - "0") @ integer_plus_maj

            graph_fractional_one = (
                pynutil.insert('fractional_part: "')
                + two_digits_fractional_part @ pynini.cross("1", "un")
                + pynutil.insert('"')
            )

            graph_fractional = (
                two_digits_fractional_part
                @ (pynini.closure(DAMO_DIGIT, 1, 2) - "1")
                @ cardinal.two_digit_non_zero
            )
            graph_fractional = (
                pynutil.insert('fractional_part: "') + graph_fractional + pynutil.insert('"')
            )

            fractional_plus_min = pynini.union(
                graph_fractional + insert_space + pynutil.insert(curr_symbol) @ graph_min_plural,
                graph_fractional_one
                + insert_space
                + pynutil.insert(curr_symbol) @ graph_min_singular,
            )

            decimal_graph_with_minor_curr = (
                integer_plus_maj + pynini.cross(decimal_separator, DAMO_SPACE) + fractional_plus_min
            )
            decimal_graph_with_minor_curr |= pynutil.add_weight(
                integer_plus_maj
                + pynini.cross(decimal_separator, DAMO_SPACE)
                + pynutil.insert('fractional_part: "')
                + two_digits_fractional_part @ cardinal.two_digit_non_zero
                + pynutil.insert('"'),
                weight=0.0001,
            )

            decimal_graph_with_minor_curr |= pynutil.delete("0,") + fractional_plus_min
            decimal_graph_with_minor_curr = pynini.union(
                pynutil.delete(curr_symbol)
                + pynini.closure(delete_space, 0, 1)
                + decimal_graph_with_minor_curr
                + preserve_order,
                decimal_graph_with_minor_curr
                + preserve_order
                + pynini.closure(delete_space, 0, 1)
                + pynutil.delete(curr_symbol),
            )

            decimal_graph_with_minor = (
                decimal_graph_with_minor_curr
                if decimal_graph_with_minor is None
                else pynini.union(decimal_graph_with_minor, decimal_graph_with_minor_curr)
            )

        final_graph = graph | pynutil.add_weight(decimal_graph_with_minor, -0.001)

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
