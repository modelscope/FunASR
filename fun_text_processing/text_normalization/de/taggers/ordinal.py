# Adapted from https://github.com/google/TextNormalizationCoveringGrammars
# Russian minimally supervised number grammar.

import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_DIGIT, GraphFst
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "2." -> ordinal { integer: "zwei" } }
        "2tes" -> ordinal { integer: "zwei" } }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic=False):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph
        endings = ["ter", "tes", "tem", "te", "ten"]
        self.graph = (
            (
                pynini.closure(DAMO_DIGIT | pynini.accep("."))
                + pynutil.delete(
                    pynutil.add_weight(pynini.union(*endings), weight=0.0001) | pynini.accep(".")
                )
            )
            @ cardinal_graph
        ).optimize()
        final_graph = pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
