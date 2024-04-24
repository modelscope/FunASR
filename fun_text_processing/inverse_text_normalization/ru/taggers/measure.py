from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, e.g.
        "два килограма" -> measure { cardinal { integer: "2 кг" } }

    Args:
        tn_measure: Text normalization Cardinal graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_measure, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        tn_measure = tn_measure.tagger_graph_default @ tn_measure.verbalizer_graph
        graph = tn_measure.invert().optimize()
        graph = pynutil.insert('cardinal { integer: "') + graph + pynutil.insert('" }')
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
