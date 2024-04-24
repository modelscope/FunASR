import pynini
from fun_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space
from pynini.lib import pynutil


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelisted tokens
        e.g. misses -> tokens { name: "Mrs." }
    Args:
        tn_whitelist_tagger: TN whitelist tagger
    """

    def __init__(self, tn_whitelist_tagger: GraphFst, deterministic: bool = True):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        whitelist = pynini.invert(tn_whitelist_tagger.graph)
        graph = pynutil.insert('name: "') + convert_space(whitelist) + pynutil.insert('"')
        self.fst = graph.optimize()
