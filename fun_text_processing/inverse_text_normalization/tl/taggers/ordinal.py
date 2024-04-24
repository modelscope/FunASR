import pynini
from fun_text_processing.inverse_text_normalization.tl.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.tl.graph_utils import DAMO_CHAR, GraphFst
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. thirteenth -> ordinal { integer: "13" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_teens = pynini.string_file(get_abs_path("data/ordinals/teen.tsv"))
        graph = pynini.closure(DAMO_CHAR) + pynini.union(
            graph_digit, graph_teens, pynini.cross("tieth", "ty"), pynini.cross("th", "")
        )

        self.graph = graph @ cardinal_graph
        final_graph = pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
