import pynini
from fun_text_processing.inverse_text_normalization.ko.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.ko.graph_utils import (
    DAMO_SIGMA,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    get_singulars,
)
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. minus twelve kilograms -> measure { negative: "true" cardinal { integer: "12" } units: "kg" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        decimal_graph = decimal.final_graph_wo_negative

        unit_graph = pynini.string_file(get_abs_path("data/measurements.tsv"))

        graph_unit = pynini.invert(unit_graph)  # singular -> abbr

        ## 마이너 负
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("마이너", '"true"') + delete_extra_space,
            0,
            1,
        )

        graph_units = pynutil.insert('units: "') + graph_unit + pynutil.insert('"')

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_extra_space
            + graph_units
        )
        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + cardinal_graph
            + pynutil.insert('"')
            + pynutil.insert(" }")
            + delete_extra_space
            + graph_units
        )

        final_graph = subgraph_decimal | subgraph_cardinal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
