import pynini
from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "23 4/6" ->
    fraction { integer: "drei und zwanzig" numerator: "vier" denominator: "sechs" preserve_order: true }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph

        self.optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1
        )
        self.integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        self.numerator = (
            pynutil.insert('numerator: "')
            + cardinal_graph
            + pynini.cross(pynini.union("/", " / "), '" ')
        )
        self.denominator = pynutil.insert('denominator: "') + cardinal_graph + pynutil.insert('"')

        self.graph = (
            self.optional_graph_negative
            + pynini.closure(self.integer + pynini.accep(" "), 0, 1)
            + self.numerator
            + self.denominator
        )

        graph = self.graph + pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
