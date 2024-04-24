import pynini
from fun_text_processing.inverse_text_normalization.pt.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


def get_quantity(
    decimal: "pynini.FstLike", cardinal_up_to_million: "pynini.FstLike"
) -> "pynini.FstLike":
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. one million -> integer_part: "1" quantity: "million"
    e.g. one point five million -> integer_part: "1" fractional_part: "5" quantity: "million"

    Args:
        decimal: decimal FST
        cardinal_up_to_million: cardinal FST
    """
    numbers = cardinal_up_to_million @ (
        pynutil.delete(pynini.closure("0"))
        + pynini.difference(DAMO_DIGIT, "0")
        + pynini.closure(DAMO_DIGIT)
    )

    suffix = pynini.union(
        "milhão",
        "milhões",
        "bilhão",
        "bilhões",
        "trilhão",
        "trilhões",
        "quatrilhão",
        "quatrilhões",
        "quintilhão",
        "quintilhões",
        "sextilhão",
        "sextilhões",
    )
    res = (
        pynutil.insert('integer_part: "')
        + numbers
        + pynutil.insert('"')
        + delete_extra_space
        + pynutil.insert('quantity: "')
        + suffix
        + pynutil.insert('"')
    )
    res |= (
        decimal + delete_extra_space + pynutil.insert('quantity: "') + suffix + pynutil.insert('"')
    )
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        Decimal point is either "." or ",", determined by whether "ponto" or "vírgula" is spoken.
            e.g. menos um vírgula dois seis -> decimal { negative: "true" integer_part: "1" morphosyntactic_features: "," fractional_part: "26" }
            e.g. menos um ponto dois seis -> decimal { negative: "true" integer_part: "1" morphosyntactic_features: "." fractional_part: "26" }

        This decimal rule assumes that decimals can be pronounced as:
        (a cardinal) + ('vírgula' or 'ponto') plus (any sequence of cardinals <1000, including 'zero')

        Also writes large numbers in shortened form, e.g.
            e.g. um vírgula dois seis milhões -> decimal { negative: "false" integer_part: "1" morphosyntactic_features: "," fractional_part: "26" quantity: "milhões" }
            e.g. dois milhões -> decimal { negative: "false" integer_part: "2" quantity: "milhões" }
            e.g. mil oitcentos e vinte e quatro milhões -> decimal { negative: "false" integer_part: "1824" quantity: "milhões" }
    Args:
        cardinal: CardinalFst

    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        # number after decimal point can be any series of cardinals <1000, including 'zero'
        graph_decimal = cardinal.numbers_up_to_thousand
        graph_decimal = pynini.closure(graph_decimal + delete_space) + graph_decimal
        self.graph = graph_decimal

        # decimal point can be denoted by 'vírgula' or 'ponto'
        decimal_point = pynini.cross("vírgula", 'morphosyntactic_features: ","')
        decimal_point |= pynini.cross("ponto", 'morphosyntactic_features: "."')

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("menos", '"true"') + delete_extra_space,
            0,
            1,
        )

        graph_fractional = (
            pynutil.insert('fractional_part: "') + graph_decimal + pynutil.insert('"')
        )

        cardinal_graph = cardinal.graph_no_exception | pynini.string_file(
            get_abs_path("data/numbers/zero.tsv")
        )
        graph_integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        final_graph_wo_sign = (
            pynini.closure(graph_integer + delete_extra_space, 0, 1)
            + decimal_point
            + delete_extra_space
            + graph_fractional
        )
        final_graph = optional_graph_negative + final_graph_wo_sign

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign, cardinal.numbers_up_to_million
        )
        final_graph |= optional_graph_negative + get_quantity(
            final_graph_wo_sign, cardinal.numbers_up_to_million
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
