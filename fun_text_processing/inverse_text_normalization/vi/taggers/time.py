import pynini
from fun_text_processing.inverse_text_normalization.vi.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from fun_text_processing.inverse_text_normalization.vi.utils import get_abs_path
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. hai rưỡi -> time { hours: "2" minutes: "30" }
        e.g. chín giờ kém hai mươi -> time { hours: "8" minutes: "40" }
        e.g. ba phút hai giây -> time { minutes: "3" seconds: "2" }
        e.g. mười giờ chín phút bốn mươi lăm giây -> time { hours: "10" minutes: "9" seconds: "45" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")
        # hours, minutes, seconds, suffix, zone, style, speak_period

        graph_hours_to = pynini.string_file(get_abs_path("data/time/hours_to.tsv"))
        graph_minutes_to = pynini.string_file(get_abs_path("data/time/minutes_to.tsv"))
        graph_hours = pynini.string_file(get_abs_path("data/time/hours.tsv"))
        graph_minutes = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
        time_zone_graph = pynini.invert(pynini.string_file(get_abs_path("data/time/time_zone.tsv")))

        graph_half = pynini.cross("rưỡi", "30")
        oclock = pynini.cross("giờ", "")
        minute = pynini.cross("phút", "")
        optional_minute = pynini.closure(delete_space + minute, 0, 1)
        second = pynini.cross("giây", "")

        final_graph_hour = (
            pynutil.insert('hours: "') + graph_hours + pynutil.insert('"') + delete_space + oclock
        )
        graph_minute = graph_minutes + optional_minute
        graph_second = graph_minutes + delete_space + second
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
            + (graph_minute | graph_half)
            + pynutil.insert('"')
        )

        graph_hms = (
            final_graph_hour
            + delete_extra_space
            + pynutil.insert('minutes: "')
            + graph_minutes
            + delete_space
            + minute
            + pynutil.insert('"')
            + delete_extra_space
            + pynutil.insert('seconds: "')
            + graph_second
            + pynutil.insert('"')
        )

        graph_ms = (
            pynutil.insert('minutes: "')
            + graph_minutes
            + delete_space
            + minute
            + pynutil.insert('"')
            + delete_extra_space
            + pynutil.insert('seconds: "')
            + (graph_second | graph_half)
            + pynutil.insert('"')
        )

        graph_hours_to_component = graph_hours @ graph_hours_to
        graph_minutes_to_component = graph_minutes @ graph_minutes_to

        graph_time_to = (
            pynutil.insert('hours: "')
            + graph_hours_to_component
            + pynutil.insert('"')
            + delete_space
            + oclock
            + delete_space
            + pynutil.delete("kém")
            + delete_extra_space
            + pynutil.insert('minutes: "')
            + graph_minutes_to_component
            + pynutil.insert('"')
            + optional_minute
        )

        final_graph = (final_graph_hour | graph_hm | graph_hms) + final_time_zone_optional
        final_graph |= graph_ms
        final_graph |= graph_time_to

        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
