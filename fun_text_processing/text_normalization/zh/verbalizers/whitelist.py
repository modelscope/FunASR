import pynini
from fun_text_processing.text_normalization.zh.graph_utils import FUN_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class Whitelist(GraphFst):
    """
    tokens { whitelist: "ATM" } -> A T M
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="whitelist", kind="verbalize", deterministic=deterministic)
        remove_erhua = pynutil.delete('erhua: "') + pynutil.delete("儿") + pynutil.delete('"')
        whitelist = pynutil.delete('name: "') + pynini.closure(FUN_NOT_QUOTE) + pynutil.delete('"')
        graph = remove_erhua | whitelist
        self.fst = graph.optimize()
