import pynini
from fun_text_processing.inverse_text_normalization.vi.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone, e.g.
        telephone { number_part: "1231235678" }
        -> 1231235678
    """

    def __init__(self):
        super().__init__(name="telephone", kind="verbalize")

        number_part = (
            pynutil.delete('number_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        delete_tokens = self.delete_tokens(number_part)
        self.fst = delete_tokens.optimize()
