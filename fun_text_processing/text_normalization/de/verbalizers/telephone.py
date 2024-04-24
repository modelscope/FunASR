import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
)
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone, e.g.
        telephone { country_code: "plus neun und vierzig" number_part: "null eins eins eins null null null" }
            -> "plus neun und vierzig null eins eins eins null null null"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="verbalize", deterministic=deterministic)

        country_code = (
            pynutil.delete('country_code: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        number_part = (
            pynutil.delete('number_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        self.graph = country_code + pynini.accep(" ") + number_part

        graph = self.graph + delete_preserve_order
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
