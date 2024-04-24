import pynini
from fun_text_processing.text_normalization.zh.graph_utils import FUN_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class MathSymbol(GraphFst):
    """
    tokens { sign: "加" }  -> 加
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="sign", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete('score: "') + pynini.closure(FUN_NOT_QUOTE) + pynutil.delete('"')

        self.fst = graph.optimize()
