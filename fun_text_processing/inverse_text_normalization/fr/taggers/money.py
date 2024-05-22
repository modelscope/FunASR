import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    DAMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from fun_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money
        e.g. douze euro cinq -> money { integer_part: "12" currency: "€" fractional_part: 05}
        e.g. zéro euro cinq -> money { integer_part: "0" currency: "€" fractional_part: 05}
        e.g. cinq centimes -> money { integer_part: "0" currency: "€" fractional_part: 05}

        Note, the currency symbol seems more common for exact amounts and quantities less than 'un million'
        For 'round' quantities of >=million (milliard, billion), the symbol is dropped. This allows
        use of the 'de' preposition.
        e.g. cinq millions d'euros -> money { integer_part: "5" currency: "d'euros" fractional_part: 00}
        e.g. un milliard d'euro -> money { integer_part: "5" currency: "d'euro" fractional_part: 00}
        e.g. trois virgule trois millions d'euros -> money { integer_part: "3" currency: "d'euros" fractional_part: 3}

        Currency is included for uniform tagging.

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")
        # quantity, integer_part, fractional_part, currency

        # quantities
        cardinal_graph = cardinal.graph_no_exception
        graph_decimal = decimal.final_graph_wo_negative

        # Converts currency names to symbols
        convert_currency_major = pynini.string_file(
            get_abs_path("data/money/currency_major.tsv")
        )  # major denominations
        convert_currency_minor = pynini.string_file(
            get_abs_path("data/money/currency_minor.tsv")
        )  # minor denominations to major symbol. (e.g. 5 cents -> 0.05 $ )

        accept_all_currency = (convert_currency_major | convert_currency_minor).project(
            "input"
        )  # recognizes all currencies

        # Graphs for large round amounts ('deux billiards d'euros', 'un milliard de dollars')
        graph_de = pynini.union(
            "de ", "des ", "d'"
        )  # the use of de/d'only occurs with round amounts
        graph_currency_component_large_round_amounts = graph_de + accept_all_currency
        graph_currency_component_large_round_amounts = (
            pynutil.insert(' currency: "')
            + graph_currency_component_large_round_amounts
            + pynutil.insert('"')
        )

        graph_money_large_round_amounts = (
            graph_decimal + delete_space
        )  # graph_decimal includes tags and quantities already
        graph_money_large_round_amounts += graph_currency_component_large_round_amounts

        # For standard currency
        add_leading_zero_to_double_digit = (DAMO_DIGIT + DAMO_DIGIT) | (
            pynutil.insert("0") + DAMO_DIGIT
        )

        # Graphs integer denomination for large denominations (e.g. $)
        graph_integer_component_major = (
            pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        )
        graph_integer_component_major += delete_space

        graph_currency_component_major = (
            pynutil.insert(' currency: "') + convert_currency_major + pynutil.insert('"')
        )

        graph_decimal_component_major = (
            delete_space
            + pynutil.insert(' fractional_part: "')
            + (cardinal_graph @ add_leading_zero_to_double_digit)
            + pynutil.insert('"')
        )

        # Rare cases where 'et' will separate major and minor denominations.
        delete_minor_currency = pynini.project(convert_currency_minor, "input")
        delete_minor_currency = delete_extra_space + pynutil.delete(delete_minor_currency)

        delete_et = delete_extra_space + pynutil.delete("et")

        graph_money_major = (
            graph_integer_component_major
            + graph_currency_component_major
            + delete_et.ques
            + graph_decimal_component_major.ques
            + delete_minor_currency.ques
        )

        # For cases when only small denominations are used.
        graph_integer_component_minor = pynutil.insert('integer_part: "0"')

        graph_decimal_component_minor = (
            pynutil.insert(' fractional_part: "')
            + (cardinal_graph @ add_leading_zero_to_double_digit)
            + pynutil.insert('"')
        )
        graph_decimal_component_minor += delete_extra_space

        graph_currency_component_minor = (
            pynutil.insert(' currency: "') + convert_currency_minor + pynutil.insert('"')
        )

        graph_money_minor = (
            graph_integer_component_minor
            + graph_decimal_component_minor
            + graph_currency_component_minor
        )

        graph_money_standard_amounts = graph_money_major | graph_money_minor

        final_graph = graph_money_large_round_amounts | graph_money_standard_amounts
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
