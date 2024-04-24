import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone
        e.g. telephone { number_part: "8-913-983-56-01" } -> "8-913-983-56-01"
    """

    def __init__(self):
        super().__init__(name="telephone", kind="verbalize")

        graph = (
            pynutil.delete('number_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
