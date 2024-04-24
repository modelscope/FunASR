import pynini
from fun_text_processing.text_normalization.zh.graph_utils import FUN_DIGIT, GraphFst, insert_space
from pynini.lib import pynutil


class Fraction(GraphFst):
    """
    1/5  -> tokens { fraction { numerator: "1" denominator: "5" } }
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        numerator = pynini.closure(FUN_DIGIT, 1) + pynutil.delete("/")
        denominator = pynini.closure(FUN_DIGIT, 1)
        graph = (
            pynutil.insert('numerator: "')
            + numerator
            + pynutil.insert('"')
            + insert_space
            + pynutil.insert('denominator: "')
            + denominator
            + pynutil.insert('"')
        )

        self.fst = self.add_tokens(graph).optimize()
