import pynini
from fun_text_processing.inverse_text_normalization.en.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALNUM,
    DAMO_ALPHA,
    DAMO_DIGIT,
    GraphFst,
    insert_space,
)
from pynini.lib import pynutil


def get_serial_number(cardinal):
    """
    any alphanumerical character sequence with at least one number with length greater equal to 3
    """
    digit = pynini.compose(cardinal.graph_no_exception, DAMO_DIGIT)
    character = digit | DAMO_ALPHA
    sequence = character + pynini.closure(pynutil.delete(" ") + character, 2)
    sequence = sequence @ (pynini.closure(DAMO_ALNUM) + DAMO_DIGIT + pynini.closure(DAMO_ALNUM))
    return sequence.optimize()


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
        one two three one two three five six seven eight -> { number_part: "123-123-5678" }

    This class also support card number and IP format.
        "one two three dot one double three dot o dot four o" -> { number_part: "123.133.0.40"}

        "three two double seven three two one four three two one four three double zero five" ->
            { number_part: 3277 3214 3214 3005}

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")
        # country code, number_part, extension
        digit_to_str = (
            pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize())
            | pynini.cross("0", pynini.union("o", "oh", "zero")).optimize()
        )

        str_to_digit = pynini.invert(digit_to_str)

        double_digit = pynini.union(
            *[
                pynini.cross(
                    pynini.project(str(i) @ digit_to_str, "output")
                    + pynini.accep(" ")
                    + pynini.project(str(i) @ digit_to_str, "output"),
                    pynutil.insert("double ") + pynini.project(str(i) @ digit_to_str, "output"),
                )
                for i in range(10)
            ]
        )
        double_digit.invert()

        # to handle cases like "one twenty three"
        two_digit_cardinal = pynini.compose(cardinal.graph_no_exception, DAMO_DIGIT**2)
        double_digit_to_digit = (
            pynini.compose(double_digit, str_to_digit + pynutil.delete(" ") + str_to_digit)
            | two_digit_cardinal
        )

        single_or_double_digit = (
            pynutil.add_weight(double_digit_to_digit, -0.0001) | str_to_digit
        ).optimize()
        single_or_double_digit |= (
            single_or_double_digit
            + pynini.closure(
                pynutil.add_weight(pynutil.delete(" ") + single_or_double_digit, 0.0001)
            )
        ).optimize()

        number_part = pynini.compose(
            single_or_double_digit,
            DAMO_DIGIT**3
            + pynutil.insert("-")
            + DAMO_DIGIT**3
            + pynutil.insert("-")
            + DAMO_DIGIT**4,
        ).optimize()
        number_part = (
            pynutil.insert('number_part: "') + number_part.optimize() + pynutil.insert('"')
        )

        cardinal_option = pynini.compose(single_or_double_digit, DAMO_DIGIT ** (2, 3))

        country_code = (
            pynutil.insert('country_code: "')
            + pynini.closure(pynini.cross("plus ", "+"), 0, 1)
            + (
                (pynini.closure(str_to_digit + pynutil.delete(" "), 0, 2) + str_to_digit)
                | cardinal_option
            )
            + pynutil.insert('"')
        )

        optional_country_code = pynini.closure(
            country_code + pynutil.delete(" ") + insert_space, 0, 1
        ).optimize()
        graph = optional_country_code + number_part

        # credit card number
        space_four_digits = insert_space + DAMO_DIGIT**4
        credit_card_graph = pynini.compose(
            single_or_double_digit, DAMO_DIGIT**4 + space_four_digits**3
        ).optimize()
        graph |= (
            pynutil.insert('number_part: "') + credit_card_graph.optimize() + pynutil.insert('"')
        )

        # SSN
        ssn_graph = pynini.compose(
            single_or_double_digit,
            DAMO_DIGIT**3
            + pynutil.insert("-")
            + DAMO_DIGIT**2
            + pynutil.insert("-")
            + DAMO_DIGIT**4,
        ).optimize()
        graph |= pynutil.insert('number_part: "') + ssn_graph.optimize() + pynutil.insert('"')

        # ip
        digit_or_double = (
            pynini.closure(str_to_digit + pynutil.delete(" "), 0, 1) + double_digit_to_digit
        )
        digit_or_double |= double_digit_to_digit + pynini.closure(
            pynutil.delete(" ") + str_to_digit, 0, 1
        )
        digit_or_double |= str_to_digit + (pynutil.delete(" ") + str_to_digit) ** (0, 2)
        digit_or_double |= cardinal_option
        digit_or_double = digit_or_double.optimize()

        ip_graph = digit_or_double + (pynini.cross(" dot ", ".") + digit_or_double) ** 3

        graph |= pynutil.insert('number_part: "') + ip_graph.optimize() + pynutil.insert('"')
        graph |= (
            pynutil.insert('number_part: "')
            + pynutil.add_weight(get_serial_number(cardinal=cardinal), weight=0.0001)
            + pynutil.insert('"')
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
