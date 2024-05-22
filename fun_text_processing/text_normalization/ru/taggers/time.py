import pynini
from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from fun_text_processing.text_normalization.ru.utils import get_abs_path
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        "02:15" -> time { hours: "два часа пятнадцать минут" }

    Args:
        number_names: number_names for cardinal and ordinal numbers
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, number_names: dict, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        increment_hour_ordinal = pynini.string_file(
            get_abs_path("data/time/increment_hour_ordinal.tsv")
        )
        increment_hour_cardinal = pynini.string_file(
            get_abs_path("data/time/increment_hour_cardinal.tsv")
        )
        convert_hour = pynini.string_file(get_abs_path("data/time/time_convert.tsv"))

        number = (
            pynini.closure(pynini.cross("0", ""), 0, 1) + number_names["cardinal_names_nominative"]
        )
        hour_options = pynini.project(increment_hour_ordinal, "input")
        hour_options = hour_options | pynini.project(convert_hour, "output")

        hour_exeption_ends_with_one = pynini.union(*["01", "21"])
        hour_exeption_ends_rest = pynini.union(*["02", "03", "04", "22", "23"])
        hour_other = (
            pynini.difference(
                hour_options, pynini.union(hour_exeption_ends_with_one, hour_exeption_ends_rest)
            )
        ).optimize()

        hour = hour_exeption_ends_with_one @ number + pynutil.insert(" час")
        hour |= hour_exeption_ends_rest @ number + pynutil.insert(" часа")
        hour |= hour_other @ number + pynutil.insert(" часов")

        optional_and = pynini.closure(pynutil.insert("и "), 0, 1)
        digits = pynini.union(*[str(x) for x in range(10)])
        mins_start = pynini.union(*"012345")
        mins_options = mins_start + digits
        mins_exception_ends_with_one = mins_start + pynini.accep("1")
        mins_exception_ends_rest = pynini.difference(
            mins_start + pynini.union(*"234"), pynini.union(*["12", "13", "14"])
        )
        mins_other = pynini.difference(
            mins_options, pynini.union(mins_exception_ends_with_one, mins_exception_ends_rest)
        )

        minutes = mins_exception_ends_with_one @ number + pynutil.insert(" минута")
        minutes |= mins_exception_ends_rest @ number + pynutil.insert(" минуты")
        minutes |= mins_other @ number + pynutil.insert(" минут")
        self.minutes = minutes.optimize()
        # 17:15 -> "семнадцать часов и пятнадцать минут"
        hm = (
            pynutil.insert('hours: "')
            + hour.optimize()
            + pynutil.insert('"')
            + (
                pynini.cross(":", " ")
                + pynutil.insert('minutes: "')
                + optional_and
                + minutes.optimize()
            )
            + pynutil.insert('"')
            + pynutil.insert(" preserve_order: true")
        )
        h = pynutil.insert('hours: "') + hour + pynutil.insert('"') + pynutil.delete(":00")
        self.graph_preserve_order = (hm | h).optimize()

        # 17:15 -> "пятнадцать минут шестого"
        # Requires permutations for the correct verbalization
        self.increment_hour_ordinal = pynini.compose(
            hour_options, increment_hour_ordinal
        ).optimize()
        m_next_h = (
            pynutil.insert('hours: "')
            + self.increment_hour_ordinal
            + pynutil.insert('"')
            + pynini.cross(":", " ")
            + pynutil.insert('minutes: "')
            + minutes
            + pynutil.insert('"')
        )

        # 17:45 -> "без пятнадцати минут шесть"
        # Requires permutations for the correct verbalization
        self.mins_to_h = pynini.string_file(
            get_abs_path("data/time/minutes_to_hour.tsv")
        ).optimize()
        self.increment_hour_cardinal = pynini.compose(
            hour_options, increment_hour_cardinal
        ).optimize()
        m_to_h = (
            pynutil.insert('hours: "')
            + self.increment_hour_cardinal
            + pynutil.insert('"')
            + pynini.cross(":", " ")
            + pynutil.insert('minutes: "без ')
            + self.mins_to_h
            + pynutil.insert('"')
        )

        self.final_graph = m_next_h | self.graph_preserve_order | m_to_h
        self.fst = self.add_tokens(self.final_graph)
        self.fst = self.fst.optimize()
