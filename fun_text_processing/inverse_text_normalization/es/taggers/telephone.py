import pynini
from fun_text_processing.inverse_text_normalization.es.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
        uno dos tres uno dos tres cinco seis siete ocho -> { number_part: "123-123-5678" }.
        If 10 digits are spoken, they are grouped as 3+3+4 (eg. 123-456-7890).
        If 9 digits are spoken, they are grouped as 3+3+3 (eg. 123-456-789).
        If 8 digits are spoken, they are grouped as 4+4 (eg. 1234-5678).
        In Spanish, digits are generally spoken individually, or as 2-digit numbers,
        eg. "one twenty three" = "123",
            "twelve thirty four" = "1234".

        (we ignore more complicated cases such as "three hundred and two" or "three nines").
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")

        # create `single_digits` and `double_digits` graphs as these will be
        # the building blocks of possible telephone numbers
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))

        single_digits = pynini.invert(graph_digit).optimize() | pynini.cross("0", "cero")

        double_digits = pynini.union(
            graph_twenties,
            graph_teen,
            (graph_ties + pynutil.insert("0")),
            (graph_ties + delete_space + pynutil.delete("y") + delete_space + graph_digit),
        ).invert()

        # define `ten_digit_graph`, `nine_digit_graph`, `eight_digit_graph`
        # which accept telephone numbers spoken (1) only with single digits,
        # or (2) spoken with double digits (and sometimes single digits)

        # 10-digit option (1): all single digits
        ten_digit_graph = (
            pynini.closure(single_digits + insert_space, 3, 3)
            + pynutil.delete("-")
            + pynini.closure(single_digits + insert_space, 3, 3)
            + pynutil.delete("-")
            + pynini.closure(single_digits + insert_space, 3, 3)
            + single_digits
        )

        # 10-digit option (2): (1+2) + (1+2) + (2+2) digits
        ten_digit_graph |= (
            single_digits
            + insert_space
            + double_digits
            + insert_space
            + pynutil.delete("-")
            + single_digits
            + insert_space
            + double_digits
            + insert_space
            + pynutil.delete("-")
            + double_digits
            + insert_space
            + double_digits
        )

        # 9-digit option (1): all single digits
        nine_digit_graph = (
            pynini.closure(single_digits + insert_space, 3, 3)
            + pynutil.delete("-")
            + pynini.closure(single_digits + insert_space, 3, 3)
            + pynutil.delete("-")
            + pynini.closure(single_digits + insert_space, 2, 2)
            + single_digits
        )

        # 9-digit option (2): (1+2) + (1+2) + (1+2) digits
        nine_digit_graph |= (
            single_digits
            + insert_space
            + double_digits
            + insert_space
            + pynutil.delete("-")
            + single_digits
            + insert_space
            + double_digits
            + insert_space
            + pynutil.delete("-")
            + single_digits
            + insert_space
            + double_digits
        )

        # 8-digit option (1): all single digits
        eight_digit_graph = (
            pynini.closure(single_digits + insert_space, 4, 4)
            + pynutil.delete("-")
            + pynini.closure(single_digits + insert_space, 3, 3)
            + single_digits
        )

        # 8-digit option (2): (2+2) + (2+2) digits
        eight_digit_graph |= (
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
            ten_digit_graph,
            nine_digit_graph,
            eight_digit_graph,
        )

        number_part = (
            pynutil.insert('number_part: "') + pynini.invert(number_part) + pynutil.insert('"')
        )

        graph = number_part
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
