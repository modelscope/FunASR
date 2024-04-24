import pynini
from fun_text_processing.inverse_text_normalization.ko.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.ko.graph_utils import GraphFst, convert_space
from pynini.lib import pynutil


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelisted tokens
        e.g. misses -> tokens { name: "mrs." }
    This class has highest priority among all classifier grammars. Whitelisted tokens are defined and loaded from "data/whitelist.tsv".
    """

    def __init__(self):
        super().__init__(name="whitelist", kind="classify")

        whitelist = pynini.string_file(get_abs_path("data/whitelist.tsv")).invert()
        graph = pynutil.insert('name: "') + convert_space(whitelist) + pynutil.insert('"')
        self.fst = graph.optimize()
