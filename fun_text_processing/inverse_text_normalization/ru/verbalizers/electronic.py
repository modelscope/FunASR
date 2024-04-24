import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. electronic { username: "ab@nd.ru" } -> "ab@nd.ru"
    """

    def __init__(self):
        super().__init__(name="electronic", kind="verbalize")

        graph = (
            pynutil.delete('username: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
