import pynini
from fun_text_processing.inverse_text_normalization.vi.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        e.g. 2 phần 3 -> tokens { fraction { numerator: "2" denominator: "3" } }
        e.g. 2 trên 3 -> tokens { fraction { numerator: "2" denominator: "3" } }
        e.g. 2 chia 3 -> tokens { fraction { numerator: "2" denominator: "3" } }

    Args:
        cardinal: OrdinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")
        # integer_part # numerator # denominator

        graph_cardinal = cardinal.graph_no_exception
        graph_four = pynini.cross("tư", "4")

        numerator = pynutil.insert('numerator: "') + graph_cardinal + pynutil.insert('"')
        fraction_component = pynutil.delete(pynini.union("phần", "trên", "chia"))
        denominator = (
            pynutil.insert('denominator: "') + (graph_cardinal | graph_four) + pynutil.insert('"')
        )

        graph_fraction_component = (
            numerator + delete_space + fraction_component + delete_extra_space + denominator
        )
        self.graph_fraction_component = graph_fraction_component

        graph = graph_fraction_component
        graph = graph.optimize()
        self.final_graph_wo_negative = graph

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ")
            + pynini.cross(pynini.union("âm", "trừ"), '"true"')
            + delete_extra_space,
            0,
            1,
        )
        graph = optional_graph_negative + graph
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
