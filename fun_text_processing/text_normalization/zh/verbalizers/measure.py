import pynini
from fun_text_processing.text_normalization.zh.graph_utils import (
    FUN_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class Measure(GraphFst):
    """
    tokens { measure { cardinal: "一" } units: "千克" } } ->  一千克
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        graph = (
            pynutil.delete("cardinal {")
            + delete_space
            + pynutil.delete('integer: "')
            + pynini.closure(FUN_NOT_QUOTE)
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete("}")
            + delete_space
            + pynutil.delete('units: "')
            + pynini.closure(FUN_NOT_QUOTE)
            + pynutil.delete('"')
        )
        percent_graph = (
            pynutil.delete("decimal { ")
            + pynutil.delete('integer_part: "')
            + pynutil.insert("百分之")
            + pynini.closure(FUN_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete("}")
        )
        graph |= percent_graph
        self.fst = self.delete_tokens(graph).optimize()
