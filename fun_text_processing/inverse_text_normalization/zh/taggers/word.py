import pynini
from fun_text_processing.inverse_text_normalization.zh.graph_utils import DAMO_NOT_SPACE, GraphFst
from pynini.lib import pynutil


class WordFst(GraphFst):
    """
    Finite state transducer for classifying plain tokens, that do not belong to any special class. This can be considered as the default class.
        e.g. sleep -> tokens { name: "sleep" }
    """

    def __init__(self):
        super().__init__(name="word", kind="classify")
        word = pynutil.insert('name: "') + pynini.closure(DAMO_NOT_SPACE, 1) + pynutil.insert('"')
        self.fst = word.optimize()
