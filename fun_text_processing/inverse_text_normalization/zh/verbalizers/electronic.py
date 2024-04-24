import pynini
from fun_text_processing.inverse_text_normalization.zh.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. tokens { electronic { username: "cdf1" domain: "abc.edu" } } -> cdf1@abc.edu
    """

    def __init__(self):
        super().__init__(name="electronic", kind="verbalize")
        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        protocol = (
            pynutil.delete("protocol:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph = user_name + delete_space + pynutil.insert("@") + domain
        graph |= protocol

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
