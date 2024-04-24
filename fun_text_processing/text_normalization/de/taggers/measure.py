import pynini
from fun_text_processing.text_normalization.de.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_NON_BREAKING_SPACE,
    DAMO_SIGMA,
    GraphFst,
    convert_space,
    insert_space,
)
from pynini.examples import plurals
from pynini.lib import pynutil

unit_singular = pynini.string_file(get_abs_path("data/measure/measurements.tsv"))
suppletive = pynini.string_file(get_abs_path("data/measure/suppletive.tsv"))


def singular_to_plural():
    # plural endung n/en maskuline Nomen mit den Endungen e, ent, and, ant, ist, or
    _n = DAMO_SIGMA + pynini.union("e") + pynutil.insert("n")
    _en = (
        DAMO_SIGMA
        + pynini.union(
            "ent", "and", "ant", "ist", "or", "ion", "ik", "heit", "keit", "schaft", "tät", "ung"
        )
        + pynutil.insert("en")
    )
    _nen = DAMO_SIGMA + pynini.union("in") + (pynutil.insert("e") | pynutil.insert("nen"))
    _fremd = DAMO_SIGMA + pynini.union("ma", "um", "us") + pynutil.insert("en")
    # maskuline Nomen mit den Endungen eur, ich, ier, ig, ling, ör
    _e = DAMO_SIGMA + pynini.union("eur", "ich", "ier", "ig", "ling", "ör") + pynutil.insert("e")
    _s = DAMO_SIGMA + pynini.union("a", "i", "o", "u", "y") + pynutil.insert("s")

    graph_plural = plurals._priority_union(
        suppletive, pynini.union(_n, _en, _nen, _fremd, _e, _s), DAMO_SIGMA
    ).optimize()

    return graph_plural


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure,  e.g.
        "2,4 oz" -> measure { cardinal { integer_part: "zwei" fractional_part: "vier" units: "unzen" preserve_order: true } }
        "1 oz" -> measure { cardinal { integer: "zwei" units: "unze" preserve_order: true } }
        "1 million oz" -> measure { cardinal { integer: "eins" quantity: "million" units: "unze" preserve_order: true } }
        This class also converts words containing numbers and letters
        e.g. "a-8" —> "a acht"
        e.g. "1,2-a" —> "ein komma zwei a"

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph

        graph_unit_singular = convert_space(unit_singular)
        graph_unit_plural = graph_unit_singular @ pynini.cdrewrite(
            convert_space(suppletive), "", "[EOS]", DAMO_SIGMA
        )
        optional_graph_negative = pynini.closure("-", 0, 1)

        graph_unit_denominator = (
            pynini.cross("/", "pro") + pynutil.insert(DAMO_NON_BREAKING_SPACE) + graph_unit_singular
        )

        optional_unit_denominator = pynini.closure(
            pynutil.insert(DAMO_NON_BREAKING_SPACE) + graph_unit_denominator,
            0,
            1,
        )

        unit_plural = (
            pynutil.insert('units: "')
            + (graph_unit_plural + (optional_unit_denominator) | graph_unit_denominator)
            + pynutil.insert('"')
        )

        unit_singular_graph = (
            pynutil.insert('units: "')
            + ((graph_unit_singular + optional_unit_denominator) | graph_unit_denominator)
            + pynutil.insert('"')
        )

        subgraph_decimal = (
            decimal.fst + insert_space + pynini.closure(pynutil.delete(" "), 0, 1) + unit_plural
        )

        subgraph_cardinal = (
            (optional_graph_negative + (pynini.closure(DAMO_DIGIT) - "1")) @ cardinal.fst
            + insert_space
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + unit_plural
        )

        subgraph_cardinal |= (
            (optional_graph_negative + pynini.accep("1"))
            @ cardinal.fst
            @ pynini.cdrewrite(pynini.cross("eins", "ein"), "", "", DAMO_SIGMA)
            + insert_space
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + unit_singular_graph
        )

        subgraph_fraction = (
            fraction.fst + insert_space + pynini.closure(pynutil.delete(" "), 0, 1) + unit_plural
        )

        cardinal_dash_alpha = (
            pynutil.insert('cardinal { integer: "')
            + cardinal_graph
            + pynutil.delete("-")
            + pynutil.insert('" } units: "')
            + pynini.closure(DAMO_ALPHA, 1)
            + pynutil.insert('"')
        )

        alpha_dash_cardinal = (
            pynutil.insert('units: "')
            + pynini.closure(DAMO_ALPHA, 1)
            + pynutil.delete("-")
            + pynutil.insert('"')
            + pynutil.insert(' cardinal { integer: "')
            + cardinal_graph
            + pynutil.insert('" }')
        )

        decimal_dash_alpha = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.delete("-")
            + pynutil.insert(' } units: "')
            + pynini.closure(DAMO_ALPHA, 1)
            + pynutil.insert('"')
        )

        decimal_times = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(' } units: "')
            + pynini.union("x", "X")
            + pynutil.insert('"')
        )

        cardinal_times = (
            pynutil.insert('cardinal { integer: "')
            + cardinal_graph
            + pynutil.insert('" } units: "')
            + pynini.union("x", "X")
            + pynutil.insert('"')
        )

        alpha_dash_decimal = (
            pynutil.insert('units: "')
            + pynini.closure(DAMO_ALPHA, 1)
            + pynutil.delete("-")
            + pynutil.insert('"')
            + pynutil.insert(" decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(" }")
        )

        final_graph = (
            subgraph_decimal
            | subgraph_cardinal
            | cardinal_dash_alpha
            | alpha_dash_cardinal
            | decimal_dash_alpha
            | decimal_times
            | alpha_dash_decimal
            | subgraph_fraction
            | cardinal_times
        )
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
