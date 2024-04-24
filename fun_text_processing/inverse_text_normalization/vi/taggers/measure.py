import pynini
from fun_text_processing.inverse_text_normalization.vi.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
)
from fun_text_processing.inverse_text_normalization.vi.utils import get_abs_path
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. trừ mười hai ki lô gam -> measure { negative: "true" cardinal { integer: "12" } units: "kg" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_four = pynini.cross("tư", "4")
        graph_one = pynini.cross("mốt", "1")
        graph_half = pynini.cross("rưỡi", "5")

        graph_unit = pynini.string_file(get_abs_path("data/measurements.tsv"))
        graph_unit_singular = pynini.invert(graph_unit)  # singular -> abbr

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ")
            + pynini.cross(pynini.union("âm", "trừ"), '"true"')
            + delete_extra_space,
            0,
            1,
        )

        unit_singular = convert_space(graph_unit_singular)
        unit_misc = (
            pynutil.insert("/")
            + pynutil.delete("trên")
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

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + pynutil.insert(" }")
            + delete_extra_space
            + unit_singular
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + cardinal_graph
            + pynutil.insert('"')
            + pynutil.insert(" }")
            + delete_extra_space
            + unit_singular
        )
        fraction_graph = (
            delete_extra_space
            + pynutil.insert('fractional_part: "')
            + (graph_digit | graph_half | graph_one | graph_four)
            + pynutil.insert('"')
        )

        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + cardinal_graph
            + pynutil.insert('" }')
            + delete_extra_space
            + unit_singular
            + fraction_graph
        )
        final_graph = subgraph_decimal | subgraph_cardinal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
