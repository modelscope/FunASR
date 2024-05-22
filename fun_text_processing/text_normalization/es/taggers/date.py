import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    DAMO_SPACE,
    GraphFst,
    delete_extra_space,
)
from fun_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

articles = pynini.union("de", "del", "el", "del año")
delete_leading_zero = (pynutil.delete("0") | (DAMO_DIGIT - "0")) + DAMO_DIGIT
month_numbers = pynini.string_file(get_abs_path("data/dates/months.tsv"))


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "01.04.2010" -> date { day: "un" month: "enero" year: "dos mil diez" preserve_order: true }
        "marzo 4 2000" -> date { month: "marzo" day: "cuatro" year: "dos mil" }
        "1990-20-01" -> date { year: "mil novecientos noventa" day: "veinte" month: "enero" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        number_to_month = month_numbers.optimize()
        month_graph = pynini.project(number_to_month, "output")

        numbers = cardinal.graph
        optional_leading_zero = delete_leading_zero | DAMO_DIGIT

        # 01, 31, 1
        digit_day = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 32)]) @ numbers
        day = (pynutil.insert('day: "') + digit_day + pynutil.insert('"')).optimize()

        digit_month = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 13)])
        number_to_month = digit_month @ number_to_month

        month_name = (pynutil.insert('month: "') + month_graph + pynutil.insert('"')).optimize()
        month_number = (
            pynutil.insert('month: "') + number_to_month + pynutil.insert('"')
        ).optimize()

        # prefer cardinal over year
        year = (DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT, 1, 3)  # 90, 990, 1990
        year @= numbers
        self.year = year

        year_only = pynutil.insert('year: "') + year + pynutil.insert('"')
        year_with_articles = (
            pynutil.insert('year: "')
            + pynini.closure(articles + DAMO_SPACE, 0, 1)
            + year
            + pynutil.insert('"')
        )

        graph_dmy = (
            day
            + pynini.closure(pynutil.delete(" de"))
            + DAMO_SPACE
            + month_name
            + pynini.closure(DAMO_SPACE + year_with_articles, 0, 1)
        )

        graph_mdy = (  # English influences on language
            month_name
            + delete_extra_space
            + day
            + pynini.closure(DAMO_SPACE + year_with_articles, 0, 1)
        )

        separators = [".", "-", "/"]
        for sep in separators:
            year_optional = pynini.closure(pynini.cross(sep, DAMO_SPACE) + year_only, 0, 1)
            new_graph = day + pynini.cross(sep, DAMO_SPACE) + month_number + year_optional
            graph_dmy |= new_graph
            if not deterministic:
                new_graph = month_number + pynini.cross(sep, DAMO_SPACE) + day + year_optional
                graph_mdy |= new_graph

        dash = "-"
        day_optional = pynini.closure(pynini.cross(dash, DAMO_SPACE) + day, 0, 1)
        graph_ymd = (
            DAMO_DIGIT**4 @ year_only + pynini.cross(dash, DAMO_SPACE) + month_number + day_optional
        )

        final_graph = graph_dmy + pynutil.insert(" preserve_order: true")
        final_graph |= graph_ymd
        final_graph |= graph_mdy

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
