import pynini
from fun_text_processing.inverse_text_normalization.zh.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.zh.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil

graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).optimize()


def _get_month_graph():
    """
    Transducer for month, e.g. march -> march
    """
    month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
    return month_graph


def _get_day_graph():
    """
    Tranducer for day,
    """
    day_graph = pynini.string_file(get_abs_path("data/days.tsv"))
    return day_graph


def _get_year_graph():
    """
    Transducer for year,
    """
    year_graph = graph_digit + (graph_digit | graph_zero) ** 3
    year_graph |= graph_digit**2

    return year_graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date,
        e.g. january fifth twenty twelve -> date { month: "january" day: "5" year: "2012" preserve_order: true }
        e.g. the fifth of january twenty twelve -> date { day: "5" month: "january" year: "2012" preserve_order: true }
        e.g. twenty twenty -> date { year: "2012" preserve_order: true }

    Args:
        ordinal: OrdinalFst
    """

    def __init__(self):
        super().__init__(name="date", kind="classify")

        year_graph = _get_year_graph()

        year_graph = (
            pynutil.insert('year: "') + year_graph + pynutil.delete("年") + pynutil.insert('"')
        )

        month_graph = _get_month_graph()
        month_graph = pynutil.insert('month: "') + month_graph + pynutil.insert('"')

        day_graph = _get_day_graph()
        day_graph = pynutil.insert('day: "') + day_graph + pynutil.insert('"')

        graph_md = month_graph + pynutil.insert(" ") + day_graph
        graph_ym = year_graph + pynutil.insert(" ") + month_graph
        graph_ymd = year_graph + pynutil.insert(" ") + month_graph + pynutil.insert(" ") + day_graph

        graph_year = year_graph

        final_graph = graph_ymd | graph_ym | graph_md | graph_year
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
