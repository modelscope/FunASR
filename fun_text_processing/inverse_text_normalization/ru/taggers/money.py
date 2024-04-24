from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        "два рубля" -> money { integer_part: "2 руб." }

    Args:
        tn_money: Text normalization Money graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_money, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        graph = tn_money.final_graph
        graph = graph.invert().optimize()
        graph = pynutil.insert('integer_part: "') + graph + pynutil.insert('"')
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
