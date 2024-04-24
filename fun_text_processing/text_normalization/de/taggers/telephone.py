import pynini
from fun_text_processing.text_normalization.de.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import DAMO_DIGIT, GraphFst, insert_space
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone, which includes country code, number part and extension

    E.g
    "+49 1234-1233" -> telephone { country_code: "plus neun und vierzig" number_part: "eins zwei drei vier eins zwei drei drei" preserve_order: true }
    "(012) 1234-1233" -> telephone { country_code: "null eins zwei" number_part: "eins zwei drei vier eins zwei drei drei" preserve_order: true }
    (0**)

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        graph_zero = pynini.invert(
            pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        ).optimize()
        graph_digit_no_zero = pynini.invert(
            pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        ).optimize() | pynini.cross("1", "eins")
        graph_digit = graph_digit_no_zero | graph_zero

        numbers_with_single_digits = pynini.closure(graph_digit + insert_space) + graph_digit

        two_digit_and_zero = (DAMO_DIGIT**2 @ cardinal.two_digit_non_zero) | graph_zero
        # def add_space_after_two_digit():
        #     return pynini.closure(two_digit_and_zero + insert_space) + (
        #         two_digit_and_zero
        #     )

        country_code = pynini.closure(pynini.cross("+", "plus "), 0, 1) + two_digit_and_zero
        country_code |= (
            pynutil.delete("(")
            + graph_zero
            + insert_space
            + numbers_with_single_digits
            + pynutil.delete(")")
        )
        country_code |= graph_zero + insert_space + numbers_with_single_digits

        country_code = pynutil.insert('country_code: "') + country_code + pynutil.insert('"')

        del_separator = pynini.cross(pynini.union("-", " "), " ")
        # numbers_with_two_digits = pynini.closure(graph_digit + insert_space) + add_space_after_two_digit() + pynini.closure(insert_space + graph_digit)
        # numbers = numbers_with_two_digits + pynini.closure(del_separator + numbers_with_two_digits, 0, 1)
        numbers = numbers_with_single_digits + pynini.closure(
            del_separator + numbers_with_single_digits, 0, 1
        )
        number_length = pynini.closure((DAMO_DIGIT | pynini.union("-", " ", ")", "(")), 7)
        number_part = pynini.compose(number_length, numbers)
        number = pynutil.insert('number_part: "') + number_part + pynutil.insert('"')

        graph = country_code + pynini.accep(" ") + number
        self.graph = graph
        final_graph = self.add_tokens(self.graph + pynutil.insert(" preserve_order: true"))
        self.fst = final_graph.optimize()
