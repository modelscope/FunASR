import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_SPACE, GraphFst
from pynini.lib import pynutil


class WordFst(GraphFst):
    """
    Finite state transducer for classifying word.
        e.g. dormir -> tokens { name: "dormir" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="word", kind="classify")
        word = pynutil.insert('name: "') + pynini.closure(DAMO_NOT_SPACE, 1) + pynutil.insert('"')
        self.fst = word.optimize()
