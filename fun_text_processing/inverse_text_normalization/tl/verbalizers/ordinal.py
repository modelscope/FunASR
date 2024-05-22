import pynini
from fun_text_processing.inverse_text_normalization.tl.graph_utils import (
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing ordinal, e.g.
       ordinal { integer: "13" } -> 13th
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")
        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        convert_eleven = pynini.cross("11", "11th")
        convert_twelve = pynini.cross("12", "12th")
        convert_thirteen = pynini.cross("13", "13th")
        convert_one = pynini.cross("1", "1st")
        convert_two = pynini.cross("2", "2nd")
        convert_three = pynini.cross("3", "3rd")
        convert_rest = pynutil.insert("th", weight=0.01)

        suffix = pynini.cdrewrite(
            convert_eleven
            | convert_twelve
            | convert_thirteen
            | convert_one
            | convert_two
            | convert_three
            | convert_rest,
            "",
            "[EOS]",
            DAMO_SIGMA,
        )
        graph = graph @ suffix
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
