from fun_text_processing.text_normalization.zh.graph_utils import FUN_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class Char(GraphFst):
    """
    tokens { char: "你" } -> 你
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="char", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete('name: "') + FUN_NOT_QUOTE + pynutil.delete('"')
        self.fst = graph.optimize()
