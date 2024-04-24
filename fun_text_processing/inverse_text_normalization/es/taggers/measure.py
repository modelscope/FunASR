import pynini
from fun_text_processing.inverse_text_normalization.es.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_SIGMA,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. menos doce kilogramos -> measure { cardinal { negative: "true" integer: "12" } units: "kg" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_unit_singular = pynini.string_file(get_abs_path("data/measurements_singular.tsv"))
        graph_unit_singular = pynini.invert(graph_unit_singular)  # singular -> abbr
        graph_unit_plural = pynini.string_file(get_abs_path("data/measurements_plural.tsv"))
        graph_unit_plural = pynini.invert(graph_unit_plural)  # plural -> abbr

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("menos", '"true"') + delete_extra_space,
            0,
            1,
        )

        unit_singular = convert_space(graph_unit_singular)
        unit_plural = convert_space(graph_unit_plural)
        unit_misc = (
            pynutil.insert("/")
            + pynutil.delete("por")
            + delete_space
            + convert_space(graph_unit_singular)
        )

        unit_singular = (
            pynutil.insert('units: "')
            + (
                unit_singular
                | unit_misc
                | pynutil.add_weight(unit_singular + delete_space + unit_misc, 0.01)
            )
            + pynutil.insert('"')
        )
        unit_plural = (
            pynutil.insert('units: "')
            + (
                unit_plural
                | unit_misc
                | pynutil.add_weight(unit_plural + delete_space + unit_misc, 0.01)
            )
            + pynutil.insert('"')
        )

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + pynutil.insert(" }")
            + delete_extra_space
            + unit_plural
        )
        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + ((DAMO_SIGMA - "un" - "una" - "uno") @ cardinal_graph)
            + pynutil.insert('"')
            + pynutil.insert(" }")
            + delete_extra_space
            + unit_plural
        )
        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + (pynini.cross("un", "1") | pynini.cross("una", "1") | pynini.cross("uno", "1"))
            + pynutil.insert('"')
            + pynutil.insert(" }")
            + delete_extra_space
            + unit_singular
        )
        final_graph = subgraph_decimal | subgraph_cardinal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
