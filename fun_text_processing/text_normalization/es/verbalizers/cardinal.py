import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from fun_text_processing.text_normalization.es.graph_utils import (
    add_cardinal_apocope_fem,
    shift_cardinal_gender,
    strip_cardinal_apocope,
)
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinals
            e.g. cardinal { integer: "dos" } -> "dos"

    Args:
            deterministic: if True will provide a single transduction option,
                    for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)
        optional_sign = pynini.closure(pynini.cross('negative: "true" ', "menos "), 0, 1)
        self.optional_sign = optional_sign

        integer = pynini.closure(DAMO_NOT_QUOTE, 1)
        self.integer = pynutil.delete(' "') + integer + pynutil.delete('"')

        integer = pynutil.delete("integer:") + self.integer

        graph_masc = optional_sign + integer
        graph_fem = shift_cardinal_gender(graph_masc)

        self.graph_masc = pynini.optimize(graph_masc)
        self.graph_fem = pynini.optimize(graph_fem)

        # Adding adjustment for fem gender (choice of gender will be random)
        graph = graph_masc | graph_fem

        if not deterministic:
            # For alternate renderings when apocope is omitted (i.e. cardinal stands alone)
            graph |= strip_cardinal_apocope(graph_masc)
            # "una" will drop to "un" in unique contexts
            graph |= add_cardinal_apocope_fem(graph_fem)

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
