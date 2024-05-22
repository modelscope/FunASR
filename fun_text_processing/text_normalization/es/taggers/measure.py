import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_NON_BREAKING_SPACE,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    convert_space,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.es.graph_utils import strip_cardinal_apocope
from fun_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

unit = pynini.string_file(get_abs_path("data/measures/measurements.tsv"))
unit_plural_fem = pynini.string_file(get_abs_path("data/measures/measurements_plural_fem.tsv"))
unit_plural_masc = pynini.string_file(get_abs_path("data/measures/measurements_plural_masc.tsv"))


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure,  e.g.
        "2,4 g" -> measure { cardinal { integer_part: "dos" fractional_part: "cuatro" units: "gramos" preserve_order: true } }
        "1 g" -> measure { cardinal { integer: "un" units: "gramo" preserve_order: true } }
        "1 millón g" -> measure { cardinal { integer: "un quantity: "millón" units: "gramos" preserve_order: true } }
        e.g. "a-8" —> "a ocho"
        e.g. "1,2-a" —> "uno coma dos a"
        This class also converts words containing numbers and letters
        e.g. "a-8" —> "a ocho"
        e.g. "1,2-a" —> "uno coma dos a"


    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph

        unit_singular = unit
        unit_plural = unit_singular @ (unit_plural_fem | unit_plural_masc)

        graph_unit_singular = convert_space(unit_singular)
        graph_unit_plural = convert_space(unit_plural)

        optional_graph_negative = pynini.closure("-", 0, 1)

        graph_unit_denominator = (
            pynini.cross("/", "por") + pynutil.insert(DAMO_NON_BREAKING_SPACE) + graph_unit_singular
        )

        optional_unit_denominator = pynini.closure(
            pynutil.insert(DAMO_NON_BREAKING_SPACE) + graph_unit_denominator,
            0,
            1,
        )

        unit_plural = (
            pynutil.insert('units: "')
            + ((graph_unit_plural + optional_unit_denominator) | graph_unit_denominator)
            + pynutil.insert('"')
        )

        unit_singular_graph = (
            pynutil.insert('units: "')
            + ((graph_unit_singular + optional_unit_denominator) | graph_unit_denominator)
            + pynutil.insert('"')
        )

        subgraph_decimal = (
            decimal.fst + insert_space + pynini.closure(DAMO_SPACE, 0, 1) + unit_plural
        )

        subgraph_cardinal = (
            (optional_graph_negative + (DAMO_SIGMA - "1")) @ cardinal.fst
            + insert_space
            + pynini.closure(delete_space, 0, 1)
            + unit_plural
        )

        subgraph_cardinal |= (
            (optional_graph_negative + pynini.accep("1")) @ cardinal.fst
            + insert_space
            + pynini.closure(delete_space, 0, 1)
            + unit_singular_graph
        )

        subgraph_fraction = (
            fraction.fst + insert_space + pynini.closure(delete_space, 0, 1) + unit_singular_graph
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
            + strip_cardinal_apocope(cardinal_graph)
            + pynutil.insert('" } units: "')
            + pynini.union("x", "X")
            + pynutil.insert('"')
        )

        cardinal_dash_alpha = (
            pynutil.insert('cardinal { integer: "')
            + strip_cardinal_apocope(cardinal_graph)
            + pynutil.delete("-")
            + pynutil.insert('" } units: "')
            + pynini.closure(DAMO_ALPHA, 1)
            + pynutil.insert('"')
        )

        decimal_dash_alpha = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.delete("-")
            + pynutil.insert(' } units: "')
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
            + pynutil.insert('" } preserve_order: true')
        )

        alpha_dash_decimal = (
            pynutil.insert('units: "')
            + pynini.closure(DAMO_ALPHA, 1)
            + pynutil.delete("-")
            + pynutil.insert('"')
            + pynutil.insert(" decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(" } preserve_order: true")
        )

        final_graph = (
            subgraph_decimal
            | subgraph_cardinal
            | cardinal_dash_alpha
            | alpha_dash_cardinal
            | decimal_dash_alpha
            | subgraph_fraction
            | decimal_times
            | cardinal_times
            | alpha_dash_decimal
        )
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
