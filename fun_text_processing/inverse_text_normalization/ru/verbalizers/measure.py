import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure
        e.g. measure { cardinal { integer: "2 кг" } } -> "2 кг"
    """

    def __init__(self):
        super().__init__(name="measure", kind="verbalize")

        graph = (
            pynutil.delete(' cardinal { integer: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete("}")
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
