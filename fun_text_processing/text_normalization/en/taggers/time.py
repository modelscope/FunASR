import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    GraphFst,
    convert_space,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.en.utils import (
    augment_labels_with_punct_at_end,
    get_abs_path,
    load_labels,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        12:30 a.m. est -> time { hours: "twelve" minutes: "thirty" suffix: "a m" zone: "e s t" }
        2.30 a.m. -> time { hours: "two" minutes: "thirty" suffix: "a m" }
        02.30 a.m. -> time { hours: "two" minutes: "thirty" suffix: "a m" }
        2.00 a.m. -> time { hours: "two" suffix: "a m" }
        2 a.m. -> time { hours: "two" suffix: "a m" }
        02:00 -> time { hours: "two" }
        2:00 -> time { hours: "two" }
        10:00:05 a.m. -> time { hours: "ten" minutes: "zero" seconds: "five" suffix: "a m" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)
        suffix_labels = load_labels(get_abs_path("data/time/suffix.tsv"))
        suffix_labels.extend(augment_labels_with_punct_at_end(suffix_labels))
        suffix_graph = pynini.string_map(suffix_labels)

        time_zone_graph = pynini.string_file(get_abs_path("data/time/zone.tsv"))

        # only used for < 1000 thousand -> 0 weight
        cardinal = cardinal.graph

        labels_hour = [str(x) for x in range(0, 24)]
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        delete_leading_zero_to_double_digit = (DAMO_DIGIT + DAMO_DIGIT) | (
            pynini.closure(pynutil.delete("0"), 0, 1) + DAMO_DIGIT
        )

        graph_hour = delete_leading_zero_to_double_digit @ pynini.union(*labels_hour) @ cardinal

        graph_minute_single = pynini.union(*labels_minute_single) @ cardinal
        graph_minute_double = pynini.union(*labels_minute_double) @ cardinal

        final_graph_hour = pynutil.insert('hours: "') + graph_hour + pynutil.insert('"')
        final_graph_minute = (
            pynutil.insert('minutes: "')
            + (pynini.cross("0", "o") + insert_space + graph_minute_single | graph_minute_double)
            + pynutil.insert('"')
        )
        final_graph_second = (
            pynutil.insert('seconds: "')
            + (pynini.cross("0", "o") + insert_space + graph_minute_single | graph_minute_double)
            + pynutil.insert('"')
        )
        final_suffix = (
            pynutil.insert('suffix: "') + convert_space(suffix_graph) + pynutil.insert('"')
        )
        final_suffix_optional = pynini.closure(delete_space + insert_space + final_suffix, 0, 1)
        final_time_zone_optional = pynini.closure(
            delete_space
            + insert_space
            + pynutil.insert('zone: "')
            + convert_space(time_zone_graph)
            + pynutil.insert('"'),
            0,
            1,
        )

        # 2:30 pm, 02:30, 2:00
        graph_hm = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynutil.delete("00") | insert_space + final_graph_minute)
            + final_suffix_optional
            + final_time_zone_optional
        )

        # 10:30:05 pm,
        graph_hms = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynini.cross("00", ' minutes: "zero"') | insert_space + final_graph_minute)
            + pynutil.delete(":")
            + (pynini.cross("00", ' seconds: "zero"') | insert_space + final_graph_second)
            + final_suffix_optional
            + final_time_zone_optional
        )

        # 2.xx pm/am
        graph_hm2 = (
            final_graph_hour
            + pynutil.delete(".")
            + (pynutil.delete("00") | insert_space + final_graph_minute)
            + delete_space
            + insert_space
            + final_suffix
            + final_time_zone_optional
        )
        # 2 pm est
        graph_h = (
            final_graph_hour + delete_space + insert_space + final_suffix + final_time_zone_optional
        )
        final_graph = (graph_hm | graph_h | graph_hm2 | graph_hms).optimize()

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
