import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    GraphFst,
    delete_hyphen,
    delete_space,
    insert_space,
)
from fun_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers. Assumes conventional grouping for Metropolitan France (and overseas departments)
    (two number sequences are grouped as individual cardinals) or digit by digit (chiffre-par-chiffre) e.g.
    "zero un quatre-vingt-deux zero deux vingt-deux cinquante" -> { number_part: "01 42 02 22 50" }
    "zero un quatre deux zero deux deux deux cinq zero" -> { number_part: "01 42 02 22 50" }

    In cases where only one digit of the first pairing is admitted, assumes that the 0 was skipped.
    "une vingt-trois quatre-vingt zero six dix-sept" -> { number_part: "01 23 40 06 17" }
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")

        # create `single_digits` and `double_digits` graphs as these will be
        # the building blocks of possible telephone numbers
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_ties_unique = pynini.string_file((get_abs_path("data/numbers/ties_unique.tsv")))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        double_digits = pynini.union(
            graph_teen,
            graph_ties_unique,
            (graph_ties + pynutil.insert("0")),
            (graph_ties + delete_hyphen + graph_digit),
        )

        graph_first_pair = graph_zero + delete_space + graph_digit
        graph_first_pair |= pynutil.insert("0") + graph_digit  # if zero is omitted
        graph_first_pair += (
            delete_space + insert_space
        )  # delete_space since closure allows possible gaps to be removed

        # All digits
        single_digits = graph_digit | graph_zero

        graph_pair_all_digits = single_digits + delete_space
        graph_pair_all_digits += single_digits

        graph_all_digits = pynini.closure(graph_pair_all_digits + delete_space + insert_space, 3, 3)
        graph_all_digits = graph_first_pair + graph_all_digits + graph_pair_all_digits

        # Paired digits
        graph_pair_digits_and_ties = double_digits | graph_pair_all_digits

        graph_digits_and_ties = pynini.closure(
            graph_pair_digits_and_ties + delete_space + insert_space, 3, 3
        )
        graph_digits_and_ties = (
            graph_first_pair + graph_digits_and_ties + graph_pair_digits_and_ties
        )

        number_part = pynini.union(graph_all_digits, graph_digits_and_ties)

        number_part = pynutil.insert('number_part: "') + number_part + pynutil.insert('"')

        graph = number_part
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
