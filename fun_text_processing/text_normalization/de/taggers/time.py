import pynini
from fun_text_processing.text_normalization.de.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    GraphFst,
    convert_space,
    insert_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        "02:15 Uhr est" -> time { hours: "2" minutes: "15" zone: "e s t"}
        "2 Uhr" -> time { hours: "2" }
        "09:00 Uhr" -> time { hours: "2" }
        "02:15:10 Uhr" -> time { hours: "2" minutes: "15" seconds: "10"}

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        final_suffix = pynutil.delete(" ") + pynutil.delete("Uhr") | pynutil.delete("uhr")
        time_zone_graph = pynini.string_file(get_abs_path("data/time/time_zone.tsv"))

        labels_hour = [str(x) for x in range(0, 25)]
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        delete_leading_zero_to_double_digit = (
            pynutil.delete("0") | (DAMO_DIGIT - "0")
        ) + DAMO_DIGIT

        graph_hour = pynini.union(*labels_hour)

        graph_minute_single = pynini.union(*labels_minute_single)
        graph_minute_double = pynini.union(*labels_minute_double)

        final_graph_hour_only = pynutil.insert('hours: "') + graph_hour + pynutil.insert('"')
        final_graph_hour = (
            pynutil.insert('hours: "')
            + delete_leading_zero_to_double_digit @ graph_hour
            + pynutil.insert('"')
        )
        final_graph_minute = (
            pynutil.insert('minutes: "')
            + (pynutil.delete("0") + graph_minute_single | graph_minute_double)
            + pynutil.insert('"')
        )
        final_graph_second = (
            pynutil.insert('seconds: "')
            + (pynutil.delete("0") + graph_minute_single | graph_minute_double)
            + pynutil.insert('"')
        )
        final_time_zone_optional = pynini.closure(
            pynini.accep(" ")
            + pynutil.insert('zone: "')
            + convert_space(time_zone_graph)
            + pynutil.insert('"'),
            0,
            1,
        )

        # 02:30 Uhr
        graph_hm = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynutil.delete("00") | (insert_space + final_graph_minute))
            + final_suffix
            + final_time_zone_optional
        )

        # 10:30:05 Uhr,
        graph_hms = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynini.cross("00", ' minutes: "0"') | (insert_space + final_graph_minute))
            + pynutil.delete(":")
            + (pynini.cross("00", ' seconds: "0"') | (insert_space + final_graph_second))
            + final_suffix
            + final_time_zone_optional
            + pynutil.insert(" preserve_order: true")
        )

        # 2 Uhr est
        graph_h = final_graph_hour_only + final_suffix + final_time_zone_optional
        final_graph = (graph_hm | graph_h | graph_hms).optimize()
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
