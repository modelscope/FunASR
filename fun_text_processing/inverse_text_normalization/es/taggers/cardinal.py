import pynini
from fun_text_processing.inverse_text_normalization.es.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. menos veintitrés -> cardinal { negative: "-" integer: "23"}
    This class converts cardinals up to (but not including) "un cuatrillón",
    i.e up to "one septillion" in English (10^{24}).
    Cardinals below ten are not converted (in order to avoid
    "vivo en una casa" --> "vivo en 1 casa" and any other odd conversions.)

    Although technically Spanish grammar requires that "y" only comes after
    "10s" numbers (ie. "treinta", ..., "noventa"), these rules will convert
    numbers even with "y" in an ungrammatical place (because "y" is ignored
    inside cardinal numbers).
        e.g. "mil y una" -> cardinal { integer: "1001"}
        e.g. "ciento y una" -> cardinal { integer: "101"}
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))
        graph_hundreds = pynini.string_file(get_abs_path("data/numbers/hundreds.tsv"))

        graph_hundred_component = graph_hundreds | pynutil.insert("0")
        graph_hundred_component += delete_space
        graph_hundred_component += pynini.union(
            graph_twenties | graph_teen | pynutil.insert("00"),
            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
        )

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(DAMO_DIGIT) + (DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        graph_thousands = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("mil"),
            pynutil.insert("001") + pynutil.delete("mil"),  # because we say 'mil', not 'un mil'
            pynutil.insert("000", weight=0.1),
        )

        graph_millones = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("millones") | pynutil.delete("millón")),
            pynutil.insert("000") + pynutil.delete("millones"),  # to allow for 'mil millones'
        )

        graph_mil_millones = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("mil"),
            pynutil.insert("001") + pynutil.delete("mil"),  # because we say 'mil', not 'un mil'
        )
        graph_mil_millones += delete_space + (
            graph_millones | pynutil.insert("000") + pynutil.delete("millones")
        )  # allow for 'mil millones'
        graph_mil_millones |= pynutil.insert("000000", weight=0.1)

        # also allow 'millardo' instead of 'mil millones'
        graph_millardo = (
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("millardo") | pynutil.delete("millardos"))
        )

        graph_billones = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("billones") | pynutil.delete("billón")),
        )

        graph_mil_billones = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("mil"),
            pynutil.insert("001") + pynutil.delete("mil"),  # because we say 'mil', not 'un mil'
        )
        graph_mil_billones += delete_space + (
            graph_billones | pynutil.insert("000") + pynutil.delete("billones")
        )  # allow for 'mil billones'
        graph_mil_billones |= pynutil.insert("000000", weight=0.1)

        graph_trillones = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("trillones") | pynutil.delete("trillón")),
        )

        graph_mil_trillones = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("mil"),
            pynutil.insert("001") + pynutil.delete("mil"),  # because we say 'mil', not 'un mil'
        )
        graph_mil_trillones += delete_space + (
            graph_trillones | pynutil.insert("000") + pynutil.delete("trillones")
        )  # allow for 'mil trillones'
        graph_mil_trillones |= pynutil.insert("000000", weight=0.1)

        graph = pynini.union(
            (graph_mil_trillones | pynutil.insert("000", weight=0.1) + graph_trillones)
            + delete_space
            + (graph_mil_billones | pynutil.insert("000", weight=0.1) + graph_billones)
            + delete_space
            + pynini.union(
                graph_mil_millones,
                pynutil.insert("000", weight=0.1) + graph_millones,
                graph_millardo + graph_millones,
                graph_millardo + pynutil.insert("000", weight=0.1),
            )
            + delete_space
            + graph_thousands
            + delete_space
            + graph_hundred_component,
            graph_zero,
        )

        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0"))
            + pynini.difference(DAMO_DIGIT, "0")
            + pynini.closure(DAMO_DIGIT),
            "0",
        )

        # ignore "y" inside cardinal numbers
        graph = (
            pynini.cdrewrite(pynutil.delete("y"), DAMO_SPACE, DAMO_SPACE, DAMO_SIGMA)
            @ (DAMO_ALPHA + DAMO_SIGMA)
            @ graph
        )

        self.graph_no_exception = graph

        # save self.numbers_up_to_thousand for use in DecimalFst
        digits_up_to_thousand = DAMO_DIGIT | (DAMO_DIGIT**2) | (DAMO_DIGIT**3)
        numbers_up_to_thousand = pynini.compose(graph, digits_up_to_thousand).optimize()
        self.numbers_up_to_thousand = numbers_up_to_thousand

        # save self.numbers_up_to_million for use in DecimalFst
        digits_up_to_million = (
            DAMO_DIGIT
            | (DAMO_DIGIT**2)
            | (DAMO_DIGIT**3)
            | (DAMO_DIGIT**4)
            | (DAMO_DIGIT**5)
            | (DAMO_DIGIT**6)
        )
        numbers_up_to_million = pynini.compose(graph, digits_up_to_million).optimize()
        self.numbers_up_to_million = numbers_up_to_million

        # don't convert cardinals from zero to nine inclusive
        graph_exception = pynini.project(pynini.union(graph_digit, graph_zero), "input")

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("menos", '"-"') + DAMO_SPACE, 0, 1
        )

        final_graph = (
            optional_minus_graph + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
