import pynini
from fun_text_processing.inverse_text_normalization.pt.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date,
        e.g. primeiro de janeiro -> date { day: "1" month: "janeiro" }
        e.g. um de janeiro -> date { day: "1" month: "janeiro" }
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        digits_from_year = cardinal.digits_from_year

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))

        graph_1_to_100 = pynini.union(
            pynutil.insert("0") + graph_digit,
            graph_twenties,
            graph_teen,
            (graph_ties + pynutil.insert("0")),
            (graph_ties + pynutil.delete(" e ") + graph_digit),
        )

        digits_1_to_31 = [str("{:0>2d}").format(digits) for digits in range(1, 32)]
        graph_1_to_31 = graph_1_to_100 @ pynini.union(*digits_1_to_31)
        # can use "primeiro" for 1st day of the month
        graph_1_to_31 = pynini.union(graph_1_to_31, pynini.cross("primeiro", "01"))

        day_graph = pynutil.insert('day: "') + graph_1_to_31 + pynutil.insert('"')

        month_name_graph = pynini.string_file(get_abs_path("data/months.tsv"))
        month_name_graph = pynutil.insert('month: "') + month_name_graph + pynutil.insert('"')

        # vinte do oito -> 20/08
        digits_1_to_12 = [str("{:0>2d}").format(digits) for digits in range(1, 13)]
        graph_1_to_12 = graph_1_to_100 @ pynini.union(*digits_1_to_12)
        month_number_graph = pynutil.insert('month: "') + graph_1_to_12 + pynutil.insert('"')

        graph_dm = (
            day_graph + delete_space + pynutil.delete("de") + delete_extra_space + month_name_graph
        )

        graph_dm |= (
            day_graph
            + delete_space
            + pynutil.delete("do")
            + delete_extra_space
            + month_number_graph
            + pynutil.insert(' morphosyntactic_features: "/"')
        )

        graph_year = (
            delete_space
            + pynutil.delete("de")
            + delete_extra_space
            + pynutil.insert('year: "')
            + digits_from_year
            + pynutil.insert('"')
        )
        graph_dmy = graph_dm + graph_year.ques

        final_graph = graph_dmy
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
