import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time,
        e.g. time { hours: "à 1" minutes: "10" } -> à 1:10
        e.g. time { hours: "às 2" minutes: "45" } -> às 2:45
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        add_leading_zero_to_double_digit = (DAMO_DIGIT + DAMO_DIGIT) | (
            pynutil.insert("0") + DAMO_DIGIT
        )

        prefix = (
            pynutil.delete("morphosyntactic_features:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + delete_space
            + insert_space
        )
        optional_prefix = pynini.closure(prefix, 0, 1)

        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_DIGIT, 1)
            + pynutil.delete('"')
        )
        minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_DIGIT, 1)
            + pynutil.delete('"')
        )
        suffix = (
            delete_space
            + insert_space
            + pynutil.delete("suffix:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_suffix = pynini.closure(suffix, 0, 1)

        graph = (
            optional_prefix
            + hour
            + delete_space
            + pynutil.insert(":")
            + (minute @ add_leading_zero_to_double_digit)
            + optional_suffix
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
