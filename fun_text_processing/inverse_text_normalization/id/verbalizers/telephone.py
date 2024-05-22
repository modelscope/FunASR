import pynini
from fun_text_processing.inverse_text_normalization.id.graph_utils import DAMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone, e.g.
        telephone { number_part: "123-123-5678" }
        -> 123-123-5678
    """

    def __init__(self):
        super().__init__(name="telephone", kind="verbalize")

        number_part = (
            pynutil.delete('number_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_country_code = pynini.closure(
            pynutil.delete('country_code: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + pynini.accep(" "),
            0,
            1,
        )
        delete_tokens = self.delete_tokens(optional_country_code + number_part)
        self.fst = delete_tokens.optimize()
