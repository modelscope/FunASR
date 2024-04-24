import pynini
from fun_text_processing.inverse_text_normalization.pt.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
        um dois um dois três quatro cinco seis sete oito nove -> { number_part: "(12) 12345-6789" }.
        If 11 digits are spoken, they are grouped as 2+5+4 (eg. (12) 34567-8901).
        If 10 digits are spoken, they are grouped as 2+4+4 (eg. (12) 3456-7890).
        If 9 digits are spoken, they are grouped as 5+4 (eg. 12345-6789).
        If 8 digits are spoken, they are grouped as 4+4 (eg. 1234-5678).
        In portuguese, digits are generally spoken individually, or as 2-digit numbers,
        eg. "trinta e quatro oitenta e dois" = "3482",
            "meia sete vinte" = "6720".
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")

        # create `single_digits` and `double_digits` graphs as these will be
        # the building blocks of possible telephone numbers
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_half = pynini.cross("meia", "6")

        graph_all_digits = pynini.union(graph_digit, graph_half, graph_zero)

        single_digits = pynini.invert(graph_all_digits).optimize()

        double_digits = (
            pynini.union(
                graph_teen | graph_twenties,
                (graph_ties + pynutil.insert("0")),
                (graph_ties + delete_space + pynutil.delete("e") + delete_space + graph_digit),
                (graph_all_digits + delete_space + graph_all_digits),
            )
            .invert()
            .optimize()
        )

        # define `eleven_digit_graph`, `ten_digit_graph`, `nine_digit_graph`, `eight_digit_graph`
        # which accept telephone numbers spoken (1) only with single digits,
        # or (2) spoken with double digits (and sometimes single digits)

        # 11-digit option (2): (2) + (1+2+2) + (2+2) digits
        eleven_digit_graph = (
            pynutil.delete("(")
            + double_digits
            + insert_space
            + pynutil.delete(") ")
            + single_digits
            + insert_space
            + double_digits
            + insert_space
            + double_digits
            + insert_space
            + pynutil.delete("-")
            + double_digits
            + insert_space
            + double_digits
        )

        # 10-digit option (2): (2) + (2+2) + (2+2) digits
        ten_digit_graph = (
            pynutil.delete("(")
            + double_digits
            + insert_space
            + pynutil.delete(") ")
            + double_digits
            + insert_space
            + double_digits
            + insert_space
            + pynutil.delete("-")
            + double_digits
            + insert_space
            + double_digits
        )

        # 9-digit option (2): (1+2+2) + (2+2) digits
        nine_digit_graph = (
            single_digits
            + insert_space
            + double_digits
            + insert_space
            + double_digits
            + insert_space
            + pynutil.delete("-")
            + double_digits
            + insert_space
            + double_digits
        )

        # 8-digit option (2): (2+2) + (2+2) digits
        eight_digit_graph = (
            double_digits
            + insert_space
            + double_digits
            + insert_space
            + pynutil.delete("-")
            + double_digits
            + insert_space
            + double_digits
        )

        number_part = pynini.union(
            eleven_digit_graph, ten_digit_graph, nine_digit_graph, eight_digit_graph
        )

        number_part = (
            pynutil.insert('number_part: "') + pynini.invert(number_part) + pynutil.insert('"')
        )

        graph = number_part
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
