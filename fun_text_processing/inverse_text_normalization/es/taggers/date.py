import pynini
from fun_text_processing.inverse_text_normalization.es.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date,
        e.g. primero de enero -> date { day: "1" month: "enero" }
        e.g. uno de enero -> date { day: "1" month: "enero" }
    """

    def __init__(self):
        super().__init__(name="date", kind="classify")

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))

        graph_1_to_100 = pynini.union(
            graph_digit,
            graph_twenties,
            graph_teen,
            (graph_ties + pynutil.insert("0")),
            (graph_ties + pynutil.delete(" y ") + graph_digit),
        )

        digits_1_to_31 = [str(digits) for digits in range(1, 32)]
        graph_1_to_31 = graph_1_to_100 @ pynini.union(*digits_1_to_31)
        # can use "primero" for 1st day of the month
        graph_1_to_31 = pynini.union(graph_1_to_31, pynini.cross("primero", "1"))

        day_graph = pynutil.insert('day: "') + graph_1_to_31 + pynutil.insert('"')

        month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
        month_graph = pynutil.insert('month: "') + month_graph + pynutil.insert('"')

        graph_dm = (
            day_graph + delete_space + pynutil.delete("de") + delete_extra_space + month_graph
        )

        final_graph = graph_dm
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
