import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from fun_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst
from pynini.lib import pynutil


class RomanFst(GraphFst):
    """
    Finite state transducer for verbalizing roman numerals
        e.g. tokens { roman { integer: "one" } } -> one

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="roman", kind="verbalize", deterministic=deterministic)
        suffix = OrdinalFst().suffix

        cardinal = pynini.closure(DAMO_NOT_QUOTE)
        ordinal = pynini.compose(cardinal, suffix)

        graph = (
            pynutil.delete('key_cardinal: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + pynini.accep(" ")
            + pynutil.delete('integer: "')
            + cardinal
            + pynutil.delete('"')
        ).optimize()

        graph |= (
            pynutil.delete('default_cardinal: "default" integer: "')
            + cardinal
            + pynutil.delete('"')
        ).optimize()

        graph |= (
            pynutil.delete('default_ordinal: "default" integer: "') + ordinal + pynutil.delete('"')
        ).optimize()

        graph |= (
            pynutil.delete('key_the_ordinal: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + pynini.accep(" ")
            + pynutil.delete('integer: "')
            + pynini.closure(pynutil.insert("the "), 0, 1)
            + ordinal
            + pynutil.delete('"')
        ).optimize()

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
