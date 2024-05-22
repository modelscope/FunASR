import pynini
from fun_text_processing.text_normalization.zh.graph_utils import (
    FUN_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class Money(GraphFst):
    """
    tokens { money { integer_part: "一点五" fractional_part: "元" } } ->  一点五元
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        cur = (
            pynutil.delete('fractional_part: "')
            + pynini.closure(FUN_NOT_QUOTE)
            + pynutil.delete('"')
        )
        num = (
            pynutil.delete('integer_part: "')
            + pynini.closure(FUN_NOT_QUOTE)
            + pynutil.delete('"')
            + delete_space
        )
        graph = num + cur

        self.fst = self.delete_tokens(graph).optimize()
