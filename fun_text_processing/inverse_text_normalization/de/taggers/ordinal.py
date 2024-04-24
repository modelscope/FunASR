import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. dreizehnter -> tokens { name: "13." }

    Args:
        itn_cardinal_tagger: ITN Cardinal Tagger
        tn_ordinal_verbalizer: TN Ordinal Verbalizer
    """

    def __init__(
        self,
        itn_cardinal_tagger: GraphFst,
        tn_ordinal_verbalizer: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        tagger = tn_ordinal_verbalizer.graph.invert().optimize()

        graph = (
            pynutil.delete('integer: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        ) @ itn_cardinal_tagger.graph

        final_graph = tagger @ graph + pynutil.insert(".")

        graph = pynutil.insert('name: "') + final_graph + pynutil.insert('"')
        self.fst = graph.optimize()
