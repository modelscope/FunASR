import pynini
from fun_text_processing.inverse_text_normalization.ko.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.ko.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil

graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).optimize()
graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
graph_digit_inh = pynini.string_file(
    get_abs_path("data/numbers/digit_inherent_digit.tsv")
).optimize()


def _get_month_graph():
    """
    Transducer for month, e.g. march -> march
    """
    month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
    # print(month_graph)
    return month_graph


def _get_day_graph():
    """
    Transducer for month, e.g. march -> march
    """
    day_graph_num = pynini.string_file(get_abs_path("data/day.tsv"))
    day_graph_inh = pynini.string_file(get_abs_path("data/day_inherent.tsv"))
    day_graph = pynini.union(day_graph_num, day_graph_inh)
    # print(day_graph)
    return day_graph


def _get_year_graph():
    """
    Transducer for year, e.g. twenty twenty -> 2020
    """
    digit = graph_digit | graph_digit_inh
    zero = graph_zero
    year_graph_4num = digit + (digit | zero) ** 3
    year_graph_2num = digit**2

    year_graph = pynini.union(year_graph_4num, year_graph_2num)
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

        year_graph = _get_year_graph() + pynini.accep("년")
        YEAR_WEIGHT = 0.001
        year_graph = (
            pynutil.insert('year: "')
            + pynutil.add_weight(year_graph, YEAR_WEIGHT)
            + pynutil.insert('"')
        )
        # year_graph_space = pynutil.insert("year: \"") + pynutil.add_weight(year_graph, YEAR_WEIGHT) + pynutil.insert("\"") + pynutil.insert(" ")
        # year_graph = pynutil.insert("year: \"") + year_graph + pynutil.insert("\"")

        MONTH_WEIGHT = -0.001
        month_graph = _get_month_graph() + pynini.cross("", "월")
        # month_graph = pynutil.insert("month: \"") + pynutil.add_weight(month_graph, MONTH_WEIGHT) + pynutil.insert("\"")
        month_graph = pynutil.insert('month: "') + month_graph + pynutil.insert('"')
        # month_graph_space = pynutil.insert("month: \"") + month_graph + pynutil.insert("\"") + pynutil.insert(" ")

        day_graph = _get_day_graph() + pynini.cross("", "일")
        DAY_WEIGHT = -0.7
        # day_graph = pynutil.insert("day: \"") + pynutil.add_weight(day_graph, DAY_WEIGHT) + pynutil.insert("\"")
        day_graph = pynutil.insert('day: "') + day_graph + pynutil.insert('"')
        # day_graph_space = pynutil.insert("day: \"") + day_graph + pynutil.insert("\"") + pynutil.insert(" ")

        graph_ymd = year_graph + delete_space + month_graph + delete_space + day_graph
        graph_md = month_graph + delete_space + day_graph
        graph_ym = year_graph + delete_space + month_graph

        final_graph = graph_ymd | graph_md | graph_ym | year_graph | month_graph | day_graph

        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
