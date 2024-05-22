import pynini
from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: email addresses, etc.
        e.g. c d f eins at a b c punkt e d u -> tokens { name: "cdf1.abc.edu" }

    Args:
        tn_electronic_tagger: TN eletronic tagger
        tn_electronic_verbalizer: TN eletronic verbalizer
    """

    def __init__(
        self,
        tn_electronic_tagger: GraphFst,
        tn_electronic_verbalizer: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        tagger = pynini.invert(tn_electronic_verbalizer.graph).optimize()
        verbalizer = pynini.invert(tn_electronic_tagger.graph).optimize()
        final_graph = tagger @ verbalizer

        graph = pynutil.insert('name: "') + final_graph + pynutil.insert('"')
        self.fst = graph.optimize()
