from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone, e.g.
        "восемь девятьсот тринадцать девятьсот восемьдесят три пятьдесят шесть ноль один" -> telephone { number_part: "8-913-983-56-01" }

    Args:
        tn_telephone: Text normalization telephone graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_telephone: GraphFst, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        tn_telephone = tn_telephone.final_graph
        graph = tn_telephone.invert().optimize()
        graph = pynutil.insert('number_part: "') + graph + pynutil.insert('"')
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
