import pynini
from fun_text_processing.inverse_text_normalization.vi.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
)
from fun_text_processing.inverse_text_normalization.vi.utils import get_abs_path
from pynini.lib import pynutil

graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv")).optimize()
graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).optimize()
ties_graph = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).optimize()


def _get_month_graph():
    """
    Transducer for month, e.g. march -> march
    """
    month_graph = pynini.string_file(get_abs_path("data/months.tsv")).optimize()
    return month_graph


def _get_ties_graph():
    """
    Transducer for 20-99 e.g
    hai ba -> 23
    """
    graph_one = pynini.cross("mốt", "1")
    graph_four = pynini.cross("tư", "4")
    graph_five = pynini.cross("lăm", "5")
    graph_ten = pynini.cross("mươi", "")
    optional_ten = pynini.closure(delete_space + graph_ten, 0, 1)

    graph = pynini.union(
        ties_graph
        + optional_ten
        + delete_space
        + (graph_digit | graph_one | graph_four | graph_five),
        ties_graph + delete_space + graph_ten + pynutil.insert("0"),
    )
    return graph


def _get_year_graph():
    """
    Transducer for year, e.g. hai không hai mươi -> 2020
    """

    def _get_digits_graph():
        zero = pynini.cross((pynini.union("linh", "lẻ")), "0")
        four = pynini.cross("tư", "4")
        graph = pynini.union(
            zero + delete_space + (graph_digit | four),
            graph_zero + delete_space + graph_digit,
        )
        graph.optimize()
        return graph

    def _get_hundreds_graph(graph_ties, graph_digits):
        graph = (
            graph_digit
            + delete_space
            + pynutil.delete("trăm")
            + delete_space
            + (graph_teen | graph_ties | graph_digits)
        )
        return graph

    def _get_thousands_graph(graph_ties, graph_digits):
        graph_hundred_component = (
            (graph_digit | graph_zero) + delete_space + pynutil.delete("trăm")
        ) | pynutil.insert("0")
        graph = (
            graph_digit
            + delete_space
            + pynutil.delete(pynini.union("nghìn", "ngàn"))
            + delete_space
            + graph_hundred_component
            + delete_space
            + (graph_teen | graph_ties | graph_digits)
        )
        return graph

    graph_ties = _get_ties_graph()
    graph_digits = _get_digits_graph()
    graph_hundreds = _get_hundreds_graph(graph_ties, graph_digits)
    graph_thousands = _get_thousands_graph(graph_ties, graph_digits)
    year_graph = (
        # 20 19, 40 12, 2012, 2 0 0 5, 2 0 17, 938 - assuming no limit on the year
        graph_digit
        + delete_space
        + (graph_digit | graph_zero)
        + delete_space
        + (graph_teen | graph_ties | graph_digits)
        | graph_thousands
        | graph_hundreds
        | (
            graph_digit
            + pynutil.insert("0")
            + delete_space
            + (graph_ties | graph_digits | graph_teen)
        )
    )
    year_graph.optimize()
    return year_graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date,
        e.g. mười lăm tháng một năm hai nghìn mười hai -> date { day: "15" month: "1" year: "2012" preserve_order: true }
        e.g. ngày ba mốt tháng mười hai năm một chín chín chín -> date { day: "31" month: "12" year: "2012" preserve_order: true }
        e.g. năm hai không hai mốt -> date { year: "2021" preserve_order: true }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        year_graph = _get_year_graph()
        YEAR_WEIGHT = 0.001
        year_graph = pynutil.add_weight(year_graph, YEAR_WEIGHT)
        month_graph = _get_month_graph()

        month_graph = pynutil.insert('month: "') + month_graph + pynutil.insert('"')
        month_exception = pynini.project(pynini.cross("năm", "5"), "input")
        month_graph_exception = (
            pynini.project(month_graph, "input") - month_exception.arcsort()
        ) @ month_graph

        day_graph = pynutil.insert('day: "') + cardinal_graph + pynutil.insert('"')
        # day_suffix = pynini.union("ngày", "mùng")
        # optional_day = pynini.closure(day_suffix + delete_space, 0, 1)

        graph_month = pynutil.delete("tháng") + delete_space + month_graph_exception
        graph_year = (
            delete_extra_space
            + pynutil.delete("năm")
            + delete_extra_space
            + pynutil.insert('year: "')
            + pynutil.add_weight(year_graph, -YEAR_WEIGHT)
            + pynutil.insert('"')
        )
        optional_graph_year = pynini.closure(graph_year, 0, 1)
        graph_my = pynutil.delete("tháng") + delete_space + month_graph + graph_year
        graph_dmy = (
            day_graph
            + delete_space
            + pynutil.delete("tháng")
            + delete_extra_space
            + month_graph
            + optional_graph_year
        )
        graph_year = (
            pynutil.delete("năm")
            + delete_extra_space
            + pynutil.insert('year: "')
            + year_graph
            + pynutil.insert('"')
        )

        final_graph = (graph_dmy | graph_my | graph_month | graph_year) + pynutil.insert(
            " preserve_order: true"
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
