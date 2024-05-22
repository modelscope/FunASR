import pynini
from fun_text_processing.inverse_text_normalization.vi.graph_utils import GraphFst, delete_space
from fun_text_processing.inverse_text_normalization.vi.utils import get_abs_path
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. thứ nhất -> ordinal { integer: "1" }
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="classify")

        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_ordinal = pynini.cross("thứ", "")
        graph = graph_digit

        self.graph = graph
        final_graph = (
            pynutil.insert('integer: "')
            + graph_ordinal
            + delete_space
            + self.graph
            + pynutil.insert('"')
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
