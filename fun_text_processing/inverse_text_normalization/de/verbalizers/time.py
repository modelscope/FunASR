import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "8" minutes: "30" zone: "e s t" } -> 08:30 Uhr est
        time { hours: "8" } -> 8 Uhr
        time { hours: "8" minutes: "30" seconds: "10" } -> 08:30:10 Uhr
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        add_leading_zero_to_double_digit = (DAMO_DIGIT + DAMO_DIGIT) | (
            pynutil.insert("0") + DAMO_DIGIT
        )
        hour = pynutil.delete('hours: "') + pynini.closure(DAMO_DIGIT, 1) + pynutil.delete('"')
        minute = pynutil.delete('minutes: "') + pynini.closure(DAMO_DIGIT, 1) + pynutil.delete('"')

        second = pynutil.delete('seconds: "') + pynini.closure(DAMO_DIGIT, 1) + pynutil.delete('"')
        zone = (
            pynutil.delete('zone: "')
            + pynini.closure(DAMO_ALPHA + delete_space)
            + DAMO_ALPHA
            + pynutil.delete('"')
        )
        optional_zone = pynini.closure(pynini.accep(" ") + zone, 0, 1)
        graph = (
            delete_space
            + pynutil.insert(":")
            + (minute @ add_leading_zero_to_double_digit)
            + pynini.closure(
                delete_space + pynutil.insert(":") + (second @ add_leading_zero_to_double_digit),
                0,
                1,
            )
            + pynutil.insert(" Uhr")
            + optional_zone
        )
        graph_h = hour + pynutil.insert(" Uhr") + optional_zone
        graph_hm = hour @ add_leading_zero_to_double_digit + graph
        graph_hms = hour @ add_leading_zero_to_double_digit + graph
        final_graph = graph_hm | graph_hms | graph_h
        self.fst = self.delete_tokens(final_graph).optimize()
