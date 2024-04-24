import pynini
from fun_text_processing.inverse_text_normalization.ja.graph_utils import (
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
        # convert_rest = pynutil.insert("第", weight=0.01)
        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynutil.insert("第", weight=0.01)
            # + DAMO_NOT_QUOTE
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        # convert_hundred = pynini.cross("第百", "第100")
        # convert_eleven = pynini.cross("11", "十一")
        # convert_twelve = pynini.cross("12", "十二")
        # convert_thirteen = pynini.cross("13", "第十三")
        # convert_one = pynini.cross("1", "第一")
        # convert_two = pynini.cross("2", "第二")
        # convert_three = pynini.cross("3", "第三")

        # suffix = pynini.cdrewrite(
        #     convert_hundred
        # #     convert_eleven
        # #     | convert_twelve
        # #     | convert_thirteen
        # #     | convert_one
        # #     | convert_two
        # #     | convert_three,
        #     "",
        #     "[EOS]",
        #     DAMO_SIGMA,
        # )
        # graph = graph @ suffix
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
