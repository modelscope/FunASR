import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_SIGMA, GraphFst, insert_space
from fun_text_processing.text_normalization.es.graph_utils import ones
from fun_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
        123-123-5678 -> { number_part: "uno dos tres uno dos tres cinco seis siete ocho" }.
        In Spanish, digits are generally read individually, or as 2-digit numbers,
        eg. "123" = "uno dos tres",
            "1234" = "doce treinta y cuatro".
        This will verbalize sequences of 10 (3+3+4 e.g. 123-456-7890).
        9 (3+3+3 e.g. 123-456-789) and 8 (4+4 e.g. 1234-5678) digits.

        (we ignore more complicated cases such as "doscientos y dos" or "tres nueves").

    Args:
                deterministic: if True will provide a single transduction option,
                        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify")

        # create `single_digits` and `double_digits` graphs as these will be
        # the building blocks of possible telephone numbers
        single_digits = pynini.invert(graph_digit).optimize() | pynini.cross("0", "cero")

        double_digits = pynini.union(
            graph_twenties,
            graph_teen,
            (graph_ties + pynutil.delete("0")),
            (graph_ties + insert_space + pynutil.insert("y") + insert_space + graph_digit),
        )
        double_digits = pynini.invert(double_digits)

        # define `ten_digit_graph`, `nine_digit_graph`, `eight_digit_graph`
        # which produces telephone numbers spoken (1) only with single digits,
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

        # 9-digit option (1): all single digits
        nine_digit_graph = (
            pynini.closure(single_digits + insert_space, 3, 3)
            + pynutil.delete("-")
            + pynini.closure(single_digits + insert_space, 3, 3)
            + pynutil.delete("-")
            + pynini.closure(single_digits + insert_space, 2, 2)
            + single_digits
        )

        # 8-digit option (1): all single digits
        eight_digit_graph = (
            pynini.closure(single_digits + insert_space, 4, 4)
            + pynutil.delete("-")
            + pynini.closure(single_digits + insert_space, 3, 3)
            + single_digits
        )

        if not deterministic:
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

        number_part = pynini.union(ten_digit_graph, nine_digit_graph, eight_digit_graph)
        number_part @= pynini.cdrewrite(pynini.cross(ones, "uno"), "", "", DAMO_SIGMA)

        number_part = pynutil.insert('number_part: "') + number_part + pynutil.insert('"')

        graph = number_part
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
