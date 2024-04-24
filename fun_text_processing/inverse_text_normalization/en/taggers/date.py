import pynini
from fun_text_processing.inverse_text_normalization.en.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil

graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv")).optimize()
graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
ties_graph = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).optimize()


def _get_month_graph():
    """
    Transducer for month, e.g. march -> march
    """
    month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
    return month_graph


def _get_ties_graph():
    """
    Transducer for 20-99 e.g
    twenty three -> 23
    """
    graph = ties_graph + (delete_space + graph_digit | pynutil.insert("0"))
    return graph


def _get_range_graph():
    """
    Transducer for decades (1**0s, 2**0s), centuries (2*00s, 1*00s), millennia (2000s)
    """
    graph_ties = _get_ties_graph()
    graph = (graph_ties | graph_teen) + delete_space + pynini.cross("hundreds", "00s")
    graph |= pynini.cross("two", "2") + delete_space + pynini.cross("thousands", "000s")
    graph |= (
        (graph_ties | graph_teen)
        + delete_space
        + (pynini.closure(DAMO_ALPHA, 1) + (pynini.cross("ies", "y") | pynutil.delete("s")))
        @ (graph_ties | pynini.cross("ten", "10"))
        + pynutil.insert("s")
    )
    graph @= pynini.union("1", "2") + DAMO_DIGIT + DAMO_DIGIT + DAMO_DIGIT + "s"
    return graph


def _get_year_graph():
    """
    Transducer for year, e.g. twenty twenty -> 2020
    """

    def _get_digits_graph():
        zero = pynini.cross((pynini.accep("oh") | pynini.accep("o")), "0")
        graph = zero + delete_space + graph_digit
        graph.optimize()
        return graph

    def _get_thousands_graph():
        graph_ties = _get_ties_graph()
        graph_hundred_component = (
            graph_digit + delete_space + pynutil.delete("hundred")
        ) | pynutil.insert("0")
        graph = (
            graph_digit
            + delete_space
            + pynutil.delete("thousand")
            + delete_space
            + graph_hundred_component
            + delete_space
            + (graph_teen | graph_ties)
        )
        return graph

    graph_ties = _get_ties_graph()
    graph_digits = _get_digits_graph()
    graph_thousands = _get_thousands_graph()
    year_graph = (
        # 20 19, 40 12, 2012 - assuming no limit on the year
        (graph_teen + delete_space + (graph_ties | graph_digits | graph_teen))
        | (graph_ties + delete_space + (graph_ties | graph_digits | graph_teen))
        | graph_thousands
    )
    year_graph.optimize()
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

    def __init__(self, ordinal: GraphFst):
        super().__init__(name="date", kind="classify")

        ordinal_graph = ordinal.graph
        year_graph = _get_year_graph()
        YEAR_WEIGHT = 0.001
        year_graph = pynutil.add_weight(year_graph, YEAR_WEIGHT)
        month_graph = _get_month_graph()

        month_graph = pynutil.insert('month: "') + month_graph + pynutil.insert('"')

        day_graph = (
            pynutil.insert('day: "') + pynutil.add_weight(ordinal_graph, -0.7) + pynutil.insert('"')
        )
        graph_year = (
            delete_extra_space
            + pynutil.insert('year: "')
            + pynutil.add_weight(year_graph, -YEAR_WEIGHT)
            + pynutil.insert('"')
        )
        optional_graph_year = pynini.closure(
            graph_year,
            0,
            1,
        )
        graph_mdy = month_graph + (
            (delete_extra_space + day_graph)
            | graph_year
            | (delete_extra_space + day_graph + graph_year)
        )
        graph_dmy = (
            pynutil.delete("the")
            + delete_space
            + day_graph
            + delete_space
            + pynutil.delete("of")
            + delete_extra_space
            + month_graph
            + optional_graph_year
        )
        graph_year = (
            pynutil.insert('year: "') + (year_graph | _get_range_graph()) + pynutil.insert('"')
        )

        final_graph = graph_mdy | graph_dmy | graph_year
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
