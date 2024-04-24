import pynini
from fun_text_processing.inverse_text_normalization.ko.taggers.cardinal import CardinalFst
from fun_text_processing.inverse_text_normalization.ko.utils import get_abs_path, num_to_word
from fun_text_processing.inverse_text_normalization.ko.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. twelve thirty -> time { hours: "12" minutes: "30" }
        e.g. twelve past one -> time { minutes: "12" hours: "1" }
        e.g. two o clock a m -> time { hours: "2" suffix: "a.m." }
        e.g. quarter to two -> time { hours: "1" minutes: "45" }
        e.g. quarter past two -> time { hours: "2" minutes: "15" }
        e.g. half past two -> time { hours: "2" minutes: "30" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")
        # hours, minutes, seconds, suffix, zone, style, speak_period

        suffix_graph = pynini.string_file(get_abs_path("data/time/time_suffix.tsv"))
        time_zone_graph = pynini.invert(pynini.string_file(get_abs_path("data/time/time_zone.tsv")))

        hour_graph = pynini.string_file(get_abs_path("data/time/hours.tsv"))
        minute_graph = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
        second_graph = pynini.string_file(get_abs_path("data/time/seconds.tsv"))

        # only used for < 1000 thousand -> 0 weight
        # cardinal = pynutil.add_weight(CardinalFst().graph_no_exception, weight=-0.7)

        graph_hour = hour_graph
        graph_minute = minute_graph
        graph_second = second_graph

        final_graph_hour = pynutil.insert('hours: "') + graph_hour + pynutil.insert('"')

        final_suffix = (
            pynutil.insert('suffix: "') + convert_space(suffix_graph) + pynutil.insert('"')
        )
        final_suffix = delete_space + insert_space + final_suffix
        final_suffix_optional = pynini.closure(final_suffix, 0, 1)
        final_time_zone_optional = pynini.closure(
            delete_space
            + insert_space
            + pynutil.insert('zone: "')
            + convert_space(time_zone_graph)
            + pynutil.insert('"'),
            0,
            1,
        )

        graph_hm = (
            final_graph_hour
            + delete_extra_space
            + pynutil.insert('minutes: "')
            + graph_minute
            + pynutil.insert('"')
        )

        graph_hms = (
            final_graph_hour
            + delete_extra_space
            + pynutil.insert('minutes: "')
            + graph_minute
            + pynutil.insert('"')
            + delete_extra_space
            + pynutil.insert('seconds: "')
            + graph_second
            + pynutil.insert('"')
        )

        graph_h = (
            final_graph_hour
            + delete_extra_space
            + pynutil.insert('minutes: "')
            + (pynutil.insert("00") | graph_minute)
            + pynutil.insert('"')
            + final_suffix
            + final_time_zone_optional
        )

        final_graph = (graph_hm | graph_hms) + final_suffix_optional + final_time_zone_optional

        final_graph |= graph_h

        final_graph = self.add_tokens(final_graph.optimize())

        self.fst = final_graph.optimize()
