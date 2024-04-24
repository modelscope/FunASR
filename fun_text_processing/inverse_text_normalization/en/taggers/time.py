import pynini
from fun_text_processing.inverse_text_normalization.en.taggers.cardinal import CardinalFst
from fun_text_processing.inverse_text_normalization.en.utils import get_abs_path, num_to_word
from fun_text_processing.text_normalization.en.graph_utils import (
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
        to_hour_graph = pynini.string_file(get_abs_path("data/time/to_hour.tsv"))
        minute_to_graph = pynini.string_file(get_abs_path("data/time/minute_to.tsv"))

        # only used for < 1000 thousand -> 0 weight
        cardinal = pynutil.add_weight(CardinalFst().graph_no_exception, weight=-0.7)

        labels_hour = [num_to_word(x) for x in range(0, 24)]
        labels_minute_single = [num_to_word(x) for x in range(1, 10)]
        labels_minute_double = [num_to_word(x) for x in range(10, 60)]

        graph_hour = pynini.union(*labels_hour) @ cardinal

        graph_minute_single = pynini.union(*labels_minute_single) @ cardinal
        graph_minute_double = pynini.union(*labels_minute_double) @ cardinal
        graph_minute_verbose = pynini.cross("half", "30") | pynini.cross("quarter", "15")
        oclock = pynini.cross(pynini.union("o' clock", "o clock", "o'clock", "oclock"), "")

        final_graph_hour = pynutil.insert('hours: "') + graph_hour + pynutil.insert('"')
        graph_minute = (
            oclock + pynutil.insert("00")
            | pynutil.delete("o") + delete_space + graph_minute_single
            | graph_minute_double
        )
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

        # five o' clock
        # two o eight, two thirty five (am/pm)
        # two pm/am
        graph_hm = (
            final_graph_hour
            + delete_extra_space
            + pynutil.insert('minutes: "')
            + graph_minute
            + pynutil.insert('"')
        )
        # 10 past four, quarter past four, half past four
        graph_m_past_h = (
            pynutil.insert('minutes: "')
            + pynini.union(graph_minute_single, graph_minute_double, graph_minute_verbose)
            + pynutil.insert('"')
            + delete_space
            + pynutil.delete("past")
            + delete_extra_space
            + final_graph_hour
        )

        graph_quarter_time = (
            pynutil.insert('minutes: "')
            + pynini.cross("quarter", "45")
            + pynutil.insert('"')
            + delete_space
            + pynutil.delete(pynini.union("to", "till"))
            + delete_extra_space
            + pynutil.insert('hours: "')
            + to_hour_graph
            + pynutil.insert('"')
        )

        graph_m_to_h_suffix_time = (
            pynutil.insert('minutes: "')
            + ((graph_minute_single | graph_minute_double).optimize() @ minute_to_graph)
            + pynutil.insert('"')
            + pynini.closure(
                delete_space + pynutil.delete(pynini.union("min", "mins", "minute", "minutes")),
                0,
                1,
            )
            + delete_space
            + pynutil.delete(pynini.union("to", "till"))
            + delete_extra_space
            + pynutil.insert('hours: "')
            + to_hour_graph
            + pynutil.insert('"')
            + final_suffix
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
        final_graph = (
            (graph_hm | graph_m_past_h | graph_quarter_time)
            + final_suffix_optional
            + final_time_zone_optional
        )
        final_graph |= graph_h
        final_graph |= graph_m_to_h_suffix_time

        final_graph = self.add_tokens(final_graph.optimize())

        self.fst = final_graph.optimize()
