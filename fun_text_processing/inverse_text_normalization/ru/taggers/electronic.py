from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic, e.g.
        "эй би собака эн ди точка ру" -> electronic { username: "ab@nd.ru" }

    Args:
        tn_electronic: Text normalization Electronic graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_electronic, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        graph = tn_electronic.final_graph
        graph = graph.invert().optimize()
        graph = pynutil.insert('username: "') + graph + pynutil.insert('"')
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
