import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    DAMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

time_zone_graph = pynini.string_file(get_abs_path("data/time/time_zone.tsv"))
suffix = pynini.string_file(get_abs_path("data/time/time_suffix.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        "02:15 est" -> time { hours: "dos" minutes: "quince" zone: "e s t"}
        "2 h" -> time { hours: "dos" }
        "9 h" -> time { hours: "nueve" }
        "02:15:10 h" -> time { hours: "dos" minutes: "quince" seconds: "diez"}

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        delete_time_delimiter = pynutil.delete(pynini.union(".", ":"))

        one = pynini.string_map([("un", "una"), ("ún", "una")])
        change_one = pynini.cdrewrite(one, "", "", DAMO_SIGMA)
        cardinal_graph = cardinal.graph @ change_one

        day_suffix = pynutil.insert('suffix: "') + suffix + pynutil.insert('"')
        day_suffix = delete_space + insert_space + day_suffix

        delete_hora_suffix = delete_space + insert_space + pynutil.delete("h")
        delete_minute_suffix = delete_space + insert_space + pynutil.delete("min")
        delete_second_suffix = delete_space + insert_space + pynutil.delete("s")

        labels_hour_24 = [
            str(x) for x in range(0, 25)
        ]  # Can see both systems. Twelve hour requires am/pm for ambiguity resolution
        labels_hour_12 = [str(x) for x in range(1, 13)]
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        delete_leading_zero_to_double_digit = (
            pynini.closure(pynutil.delete("0") | (DAMO_DIGIT - "0"), 0, 1) + DAMO_DIGIT
        )

        graph_24 = (
            pynini.closure(DAMO_DIGIT, 1, 2)
            @ delete_leading_zero_to_double_digit
            @ pynini.union(*labels_hour_24)
        )
        graph_12 = (
            pynini.closure(DAMO_DIGIT, 1, 2)
            @ delete_leading_zero_to_double_digit
            @ pynini.union(*labels_hour_12)
        )

        graph_hour_24 = graph_24 @ cardinal_graph
        graph_hour_12 = graph_12 @ cardinal_graph

        graph_minute_single = delete_leading_zero_to_double_digit @ pynini.union(
            *labels_minute_single
        )
        graph_minute_double = pynini.union(*labels_minute_double)

        graph_minute = pynini.union(graph_minute_single, graph_minute_double) @ cardinal_graph

        final_graph_hour_only_24 = (
            pynutil.insert('hours: "') + graph_hour_24 + pynutil.insert('"') + delete_hora_suffix
        )
        final_graph_hour_only_12 = (
            pynutil.insert('hours: "') + graph_hour_12 + pynutil.insert('"') + day_suffix
        )

        final_graph_hour_24 = pynutil.insert('hours: "') + graph_hour_24 + pynutil.insert('"')
        final_graph_hour_12 = pynutil.insert('hours: "') + graph_hour_12 + pynutil.insert('"')

        final_graph_minute = pynutil.insert('minutes: "') + graph_minute + pynutil.insert('"')
        final_graph_second = pynutil.insert('seconds: "') + graph_minute + pynutil.insert('"')
        final_time_zone_optional = pynini.closure(
            delete_space
            + insert_space
            + pynutil.insert('zone: "')
            + time_zone_graph
            + pynutil.insert('"'),
            0,
            1,
        )

        # 02.30 h
        graph_hm = (
            final_graph_hour_24
            + delete_time_delimiter
            + (pynutil.delete("00") | (insert_space + final_graph_minute))
            + pynini.closure(
                delete_time_delimiter
                + (pynini.cross("00", ' seconds: "0"') | (insert_space + final_graph_second)),
                0,
                1,
            )  # For seconds 2.30.35 h
            + pynini.closure(delete_hora_suffix, 0, 1)  # 2.30 is valid if unambiguous
            + final_time_zone_optional
        )

        # 2 h 30 min
        graph_hm |= (
            final_graph_hour_24
            + delete_hora_suffix
            + delete_space
            + (pynutil.delete("00") | (insert_space + final_graph_minute))
            + delete_minute_suffix
            + pynini.closure(
                delete_space
                + (pynini.cross("00", ' seconds: "0"') | (insert_space + final_graph_second))
                + delete_second_suffix,
                0,
                1,
            )  # For seconds
            + final_time_zone_optional
        )

        # 2.30 a. m. (Only for 12 hour clock)
        graph_hm |= (
            final_graph_hour_12
            + delete_time_delimiter
            + (pynutil.delete("00") | (insert_space + final_graph_minute))
            + pynini.closure(
                delete_time_delimiter
                + (pynini.cross("00", ' seconds: "0"') | (insert_space + final_graph_second)),
                0,
                1,
            )  # For seconds 2.30.35 a. m.
            + day_suffix
            + final_time_zone_optional
        )

        graph_h = (
            pynini.union(final_graph_hour_only_24, final_graph_hour_only_12)
            + final_time_zone_optional
        )  # Should always have a time indicator, else we'll pass to cardinals

        if not deterministic:
            # This includes alternate vocalization (hour menos min, min para hour), here we shift the times and indicate a `style` tag
            hour_shift_24 = pynini.invert(
                pynini.string_file(get_abs_path("data/time/hour_to_24.tsv"))
            )
            hour_shift_12 = pynini.invert(
                pynini.string_file(get_abs_path("data/time/hour_to_12.tsv"))
            )
            minute_shift = pynini.string_file(get_abs_path("data/time/minute_to.tsv"))

            graph_hour_to_24 = graph_24 @ hour_shift_24 @ cardinal_graph
            graph_hour_to_12 = graph_12 @ hour_shift_12 @ cardinal_graph

            graph_minute_to = (
                pynini.union(graph_minute_single, graph_minute_double)
                @ minute_shift
                @ cardinal_graph
            )

            final_graph_hour_to_24 = (
                pynutil.insert('hours: "') + graph_hour_to_24 + pynutil.insert('"')
            )
            final_graph_hour_to_12 = (
                pynutil.insert('hours: "') + graph_hour_to_12 + pynutil.insert('"')
            )

            final_graph_minute_to = (
                pynutil.insert('minutes: "') + graph_minute_to + pynutil.insert('"')
            )

            graph_menos = pynutil.insert(' style: "1"')
            graph_para = pynutil.insert(' style: "2"')

            final_graph_style = graph_menos | graph_para

            # 02.30 h (omitting seconds since a bit awkward)
            graph_hm |= (
                final_graph_hour_to_24
                + delete_time_delimiter
                + insert_space
                + final_graph_minute_to
                + pynini.closure(delete_hora_suffix, 0, 1)  # 2.30 is valid if unambiguous
                + final_time_zone_optional
                + final_graph_style
            )

            # 2 h 30 min
            graph_hm |= (
                final_graph_hour_to_24
                + delete_hora_suffix
                + delete_space
                + insert_space
                + final_graph_minute_to
                + delete_minute_suffix
                + final_time_zone_optional
                + final_graph_style
            )

            # 2.30 a. m. (Only for 12 hour clock)
            graph_hm |= (
                final_graph_hour_to_12
                + delete_time_delimiter
                + insert_space
                + final_graph_minute_to
                + day_suffix
                + final_time_zone_optional
                + final_graph_style
            )

        final_graph = graph_hm | graph_h
        if deterministic:
            final_graph = final_graph + pynutil.insert(" preserve_order: true")
        final_graph = final_graph.optimize()
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
