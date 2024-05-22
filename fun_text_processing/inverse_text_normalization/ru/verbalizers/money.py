import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. money { integer_part: "2 руб." } -> "2 руб."
    """

    def __init__(self):
        super().__init__(name="money", kind="verbalize")

        graph = (
            pynutil.delete('integer_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
