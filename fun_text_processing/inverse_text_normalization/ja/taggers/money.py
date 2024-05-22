import pynini
from fun_text_processing.inverse_text_normalization.ja.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.ja.graph_utils import (
    DAMO_DIGIT,
    DAMO_NOT_SPACE,
    DAMO_SIGMA,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    # get_singulars,
    insert_space,
)
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money
        e.g. twelve dollars and five cents -> money { integer_part: "12" fractional_part: 05 currency: "$" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")
        # quantity, integer_part, fractional_part, currency

        cardinal_graph = cardinal.graph_no_exception
        # add support for missing hundred (only for 3 digit numbers)
        # "one fifty" -> "one hundred fifty"
        with_hundred = pynini.compose(
            pynini.closure(DAMO_NOT_SPACE) + pynini.accep(" ") + pynutil.insert("百") + DAMO_SIGMA,
            pynini.compose(cardinal_graph, DAMO_DIGIT**3),
        )
        cardinal_graph |= with_hundred
        graph_decimal_final = decimal.final_graph_wo_negative

        unit = pynini.string_file(get_abs_path("data/currency.tsv"))
        unit_singular = pynini.invert(unit)
        unit_plural = unit_singular
        # unit_plural = get_singulars(unit_singular)

        graph_unit_singular = (
            pynutil.insert('currency: "') + convert_space(unit_singular) + pynutil.insert('"')
        )
        graph_unit_plural = (
            pynutil.insert('currency: "') + convert_space(unit_plural) + pynutil.insert('"')
        )

        add_leading_zero_to_double_digit = (DAMO_DIGIT + DAMO_DIGIT) | (
            pynutil.insert("0") + DAMO_DIGIT
        )
        # twelve dollars (and) fifty cents, zero cents
        cents_standalone = (
            pynutil.insert('fractional_part: "')
            + pynini.union(
                pynutil.add_weight(((DAMO_SIGMA - "一") @ cardinal_graph), -0.7)
                @ add_leading_zero_to_double_digit
                + delete_space
                + pynutil.delete("セント"),  # cent
                pynini.cross("一", "01") + delete_space + pynutil.delete("セント"),  # cent
            )
            + pynutil.insert('"')
        )

        optional_cents_standalone = pynini.closure(
            delete_space
            + pynini.closure(pynutil.delete("と") + delete_space, 0, 1)  # and
            + insert_space
            + cents_standalone,
            0,
            1,
        )
        # twelve dollars fifty, only after integer
        optional_cents_suffix = pynini.closure(
            delete_extra_space
            + pynutil.insert('fractional_part: "')
            + pynutil.add_weight(cardinal_graph @ add_leading_zero_to_double_digit, -0.7)
            + pynutil.insert('"'),
            0,
            1,
        )

        graph_integer = (
            pynutil.insert('integer_part: "')
            + ((DAMO_SIGMA - "一") @ cardinal_graph)
            + pynutil.insert('"')
            + delete_extra_space
            + graph_unit_plural
            + (optional_cents_standalone | optional_cents_suffix)
        )
        graph_integer |= (
            pynutil.insert('integer_part: "')
            + pynini.cross("一", "1")
            + pynutil.insert('"')
            + delete_extra_space
            + graph_unit_singular
            + (optional_cents_standalone | optional_cents_suffix)
        )
        graph_decimal = graph_decimal_final + delete_extra_space + graph_unit_plural
        graph_decimal |= pynutil.insert('currency: "$" integer_part: "0" ') + cents_standalone
        final_graph = graph_integer | graph_decimal

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
