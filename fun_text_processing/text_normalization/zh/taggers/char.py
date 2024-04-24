from fun_text_processing.text_normalization.zh.graph_utils import FUN_CHAR, GraphFst
from pynini.lib import pynutil


class Char(GraphFst):
    """
    你 -> char { name: "你" }
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="char", kind="classify", deterministic=deterministic)

        graph = pynutil.insert('name: "') + FUN_CHAR + pynutil.insert('"')
        self.fst = graph.optimize()
