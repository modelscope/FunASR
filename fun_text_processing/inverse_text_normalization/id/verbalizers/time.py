import pynini
from fun_text_processing.inverse_text_normalization.id.graph_utils import (
    DAMO_CHAR,
    DAMO_DIGIT,
    GraphFst,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "12" minutes: "30" } -> 12:30
        time { hours: "1" minutes: "12" } -> 01:12
        time { hours: "2" suffix: "a.m." } -> 02:00 a.m.
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        add_leading_zero_to_double_digit = (DAMO_DIGIT + DAMO_DIGIT) | (
            pynutil.insert("0") + DAMO_DIGIT
        )
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
            + pynini.closure(DAMO_CHAR - " ", 1)
            + pynutil.delete('"')
        )
        optional_suffix = pynini.closure(suffix, 0, 1)
        zone = (
            delete_space
            + insert_space
            + pynutil.delete("zone:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_CHAR - " ", 1)
            + pynutil.delete('"')
        )
        optional_zone = pynini.closure(zone, 0, 1)
        graph = (
            hour @ add_leading_zero_to_double_digit
            + delete_space
            + pynutil.insert(":")
            + (minute @ add_leading_zero_to_double_digit)
            + optional_suffix
            + optional_zone
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
