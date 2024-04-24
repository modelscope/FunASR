import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { day: "02.03.89" }  -> "02.03.89"
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")
        graph = pynutil.delete('day: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        delete_tokens = self.delete_tokens(graph.optimize())
        self.fst = delete_tokens.optimize()
