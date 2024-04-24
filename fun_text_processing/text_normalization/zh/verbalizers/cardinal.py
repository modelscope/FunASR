import pynini
from fun_text_processing.text_normalization.zh.graph_utils import FUN_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class Cardinal(GraphFst):
    """
    tokens { cardinal { integer: "一二三" } } -> 一二三
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete('integer: "') + pynini.closure(FUN_NOT_QUOTE) + pynutil.delete('"')

        self.fst = self.delete_tokens(graph).optimize()
