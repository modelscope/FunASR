import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time
        e.g. time { hours: "02:15" } -> "02:15"
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        hour = (
            pynutil.delete("hours: ")
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        minutes = (
            pynutil.delete("minutes: ")
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph_preserve_order = (
            pynutil.delete('hours: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )

        # for cases that require permutations for the correct verbalization
        graph_reverse_order = hour + delete_space + pynutil.insert(":") + minutes + delete_space

        graph = graph_preserve_order | graph_reverse_order
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
