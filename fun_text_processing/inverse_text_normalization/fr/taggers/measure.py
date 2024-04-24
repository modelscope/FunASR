import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    get_singulars,
)
from fun_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure. Allows for plural form for unit.
        e.g. moins onze kilogramme -> measure { negative: "true" cardinal { integer: "11" } units: "kg" }
        e.g. trois heures -> measure { cardinal { integer: "3" } units: "h" }
        e.g. demi gramme -> measure { fraction { numerator: "1" denominator: "2" } units: "g" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_prefix = pynini.string_file(get_abs_path("data/measurements/magnitudes.tsv"))
        graph_unit_singular = pynini.string_file(get_abs_path("data/measurements/measurements.tsv"))

        unit = get_singulars(graph_unit_singular) | graph_unit_singular
        unit = graph_prefix.ques + unit

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("moins", '"true"') + delete_extra_space,
            0,
            1,
        )

        unit_misc = (
            pynutil.insert("/")
            + (pynutil.delete("par") | pynutil.delete("à"))
            + delete_space
            + unit
        )

        unit = (
            pynutil.insert('units: "')
            + (unit | unit_misc | pynutil.add_weight(unit + delete_space + unit_misc, 0.01))
            + pynutil.insert('"')
        )

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + pynutil.insert(" }")
            + delete_extra_space
            + unit
        )

        subgraph_fraction = (
            pynutil.insert("fraction { ")
            + optional_graph_negative
            + fraction.final_graph_wo_negative
            + pynutil.insert(" }")
            + delete_extra_space
            + unit
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + cardinal_graph
            + pynutil.insert('"')
            + pynutil.insert(" }")
            + delete_extra_space
            + unit
        )
        final_graph = subgraph_decimal | subgraph_cardinal | subgraph_fraction
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
