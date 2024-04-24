import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal
        e.g. cardinal { integer: "23" negative: "-" } -> -23

    Args:
        tn_cardinal_verbalizer: TN cardinal verbalizer
    """

    def __init__(self, tn_cardinal_verbalizer: GraphFst, deterministic: bool = True):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)
        self.numbers = tn_cardinal_verbalizer.numbers
        optional_sign = pynini.closure(
            pynutil.delete('negative: "') + DAMO_NOT_QUOTE + pynutil.delete('" '), 0, 1
        )
        graph = optional_sign + self.numbers
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
