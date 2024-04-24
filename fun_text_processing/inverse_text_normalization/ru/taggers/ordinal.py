import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_DIGIT, GraphFst
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinals, e.g.
        "второе" -> ordinal { integer: "2" } }

    Args:
        tn_ordinal: Text normalization Ordinal graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        tn_ordinal = tn_ordinal.ordinal_numbers

        graph = tn_ordinal.invert().optimize()
        self.graph = graph

        # do not invert numbers less than 10
        graph = pynini.compose(graph, DAMO_DIGIT ** (2, ...))
        graph = pynutil.insert('integer: "') + graph + pynutil.insert('"')
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
