import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_DIGIT, GraphFst, insert_space
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
       "тысяча один" ->  cardinal { integer: "1 001" }

    Args:
        tn_cardinal: Text normalization Cardinal graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        graph = tn_cardinal.cardinal_numbers_default
        self.graph = graph.invert().optimize()

        optional_sign = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("минус ", '"-"') + insert_space, 0, 1
        )

        # do not invert numbers less than 10
        graph = pynini.compose(graph, DAMO_DIGIT ** (2, ...))
        graph = optional_sign + pynutil.insert('integer: "') + graph + pynutil.insert('"')
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
