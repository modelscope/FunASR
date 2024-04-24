import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_SIGMA,
    SINGULAR_TO_PLURAL,
    GraphFst,
    convert_space,
    insert_space,
)
from fun_text_processing.text_normalization.en.utils import get_abs_path, load_labels
from pynini.lib import pynutil

min_singular = pynini.string_file(get_abs_path("data/money/currency_minor_singular.tsv"))
min_plural = pynini.string_file(get_abs_path("data/money/currency_minor_plural.tsv"))
maj_singular = pynini.string_file((get_abs_path("data/money/currency_major.tsv")))


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g.
        $12.05 -> money { integer_part: "twelve" currency_maj: "dollars" fractional_part: "five" currency_min: "cents" preserve_order: true }
        $12.0500 -> money { integer_part: "twelve" currency_maj: "dollars" fractional_part: "five" currency_min: "cents" preserve_order: true }
        $1 -> money { currency_maj: "dollar" integer_part: "one" }
        $1.00 -> money { currency_maj: "dollar" integer_part: "one" }
        $0.05 -> money { fractional_part: "five"  currency_min: "cents" preserve_order: true }
        $1 million -> money { currency_maj: "dollars" integer_part: "one" quantity: "million" }
        $1.2 million -> money { currency_maj: "dollars" integer_part: "one"  fractional_part: "two" quantity: "million" }
        $1.2320 -> money { currency_maj: "dollars" integer_part: "one"  fractional_part: "two three two" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph_with_and
        graph_decimal_final = decimal.final_graph_wo_negative_w_abbr

        maj_singular_labels = load_labels(get_abs_path("data/money/currency_major.tsv"))
        maj_unit_plural = convert_space(maj_singular @ SINGULAR_TO_PLURAL)
        maj_unit_singular = convert_space(maj_singular)

        graph_maj_singular = (
            pynutil.insert('currency_maj: "') + maj_unit_singular + pynutil.insert('"')
        )
        graph_maj_plural = pynutil.insert('currency_maj: "') + maj_unit_plural + pynutil.insert('"')

        optional_delete_fractional_zeros = pynini.closure(
            pynutil.delete(".") + pynini.closure(pynutil.delete("0"), 1), 0, 1
        )

        graph_integer_one = (
            pynutil.insert('integer_part: "') + pynini.cross("1", "one") + pynutil.insert('"')
        )
        # only for decimals where third decimal after comma is non-zero or with quantity
        decimal_delete_last_zeros = (
            pynini.closure(DAMO_DIGIT | pynutil.delete(","))
            + pynini.accep(".")
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

        final_graph = (graph_integer_only + optional_delete_fractional_zeros) | graph_decimal

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
        # format ** dollars ** cent
        decimal_graph_with_minor = None
        integer_graph_reordered = None
        decimal_default_reordered = None
        for curr_symbol, _ in maj_singular_labels:
            preserve_order = pynutil.insert(" preserve_order: true")
            integer_plus_maj = (
                graph_integer + insert_space + pynutil.insert(curr_symbol) @ graph_maj_plural
            )
            integer_plus_maj |= (
                graph_integer_one + insert_space + pynutil.insert(curr_symbol) @ graph_maj_singular
            )

            integer_plus_maj_with_comma = pynini.compose(
                DAMO_DIGIT - "0" + pynini.closure(DAMO_DIGIT | pynutil.delete(",")),
                integer_plus_maj,
            )
            integer_plus_maj = pynini.compose(pynini.closure(DAMO_DIGIT) - "0", integer_plus_maj)
            integer_plus_maj |= integer_plus_maj_with_comma

            graph_fractional_one = two_digits_fractional_part @ pynini.cross("1", "one")
            graph_fractional_one = (
                pynutil.insert('fractional_part: "') + graph_fractional_one + pynutil.insert('"')
            )
            graph_fractional = (
                two_digits_fractional_part
                @ (pynini.closure(DAMO_DIGIT, 1, 2) - "1")
                @ cardinal.graph_hundred_component_at_least_one_none_zero_digit
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
                integer_plus_maj + pynini.cross(".", " ") + fractional_plus_min
            )

            if not deterministic:
                decimal_graph_with_minor_curr |= pynutil.add_weight(
                    integer_plus_maj
                    + pynini.cross(".", " ")
                    + pynutil.insert('fractional_part: "')
                    + two_digits_fractional_part
                    @ cardinal.graph_hundred_component_at_least_one_none_zero_digit
                    + pynutil.insert('"'),
                    weight=0.0001,
                )
                default_fraction_graph = (
                    decimal_delete_last_zeros | decimal_with_quantity
                ) @ graph_decimal_final
            decimal_graph_with_minor_curr |= (
                pynini.closure(pynutil.delete("0"), 0, 1)
                + pynutil.delete(".")
                + fractional_plus_min
            )
            decimal_graph_with_minor_curr = (
                pynutil.delete(curr_symbol) + decimal_graph_with_minor_curr + preserve_order
            )

            decimal_graph_with_minor = (
                decimal_graph_with_minor_curr
                if decimal_graph_with_minor is None
                else pynini.union(
                    decimal_graph_with_minor, decimal_graph_with_minor_curr
                ).optimize()
            )

            if not deterministic:
                integer_graph_reordered_curr = (
                    pynutil.delete(curr_symbol) + integer_plus_maj + preserve_order
                ).optimize()

                integer_graph_reordered = (
                    integer_graph_reordered_curr
                    if integer_graph_reordered is None
                    else pynini.union(
                        integer_graph_reordered, integer_graph_reordered_curr
                    ).optimize()
                )
                decimal_default_reordered_curr = (
                    pynutil.delete(curr_symbol)
                    + default_fraction_graph
                    + insert_space
                    + pynutil.insert(curr_symbol) @ graph_maj_plural
                )

                decimal_default_reordered = (
                    decimal_default_reordered_curr
                    if decimal_default_reordered is None
                    else pynini.union(decimal_default_reordered, decimal_default_reordered_curr)
                ).optimize()

        # weight for SH
        final_graph |= pynutil.add_weight(decimal_graph_with_minor, -0.0001)

        if not deterministic:
            final_graph |= integer_graph_reordered | decimal_default_reordered
            # to handle "$2.00" cases
            final_graph |= pynini.compose(
                DAMO_SIGMA + pynutil.delete(".") + pynini.closure(pynutil.delete("0"), 1),
                integer_graph_reordered,
            )
        final_graph = self.add_tokens(final_graph.optimize())
        self.fst = final_graph.optimize()
