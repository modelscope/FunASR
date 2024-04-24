import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_SIGMA,
    DAMO_SPACE,
    DAMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.es.graph_utils import cardinal_separator
from fun_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teen.tsv")))
ties = pynini.invert(pynini.string_file(get_abs_path("data/numbers/ties.tsv")))
twenties = pynini.invert(pynini.string_file(get_abs_path("data/numbers/twenties.tsv")))
hundreds = pynini.invert(pynini.string_file(get_abs_path("data/numbers/hundreds.tsv")))


def filter_punctuation(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Helper function for parsing number strings. Converts common cardinal strings (groups of three digits delineated by 'cardinal_separator' - see graph_utils)
    and converts to a string of digits:
        "1 000" -> "1000"
        "1.000.000" -> "1000000"
    Args:
        fst: Any pynini.FstLike object. Function composes fst onto string parser fst

    Returns:
        fst: A pynini.FstLike object
    """
    exactly_three_digits = DAMO_DIGIT**3  # for blocks of three
    up_to_three_digits = pynini.closure(DAMO_DIGIT, 1, 3)  # for start of string

    cardinal_string = pynini.closure(
        DAMO_DIGIT, 1
    )  # For string w/o punctuation (used for page numbers, thousand series)

    cardinal_string |= (
        up_to_three_digits
        + pynutil.delete(cardinal_separator)
        + pynini.closure(exactly_three_digits + pynutil.delete(cardinal_separator))
        + exactly_three_digits
    )

    return cardinal_string @ fst


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "1000" ->  cardinal { integer: "mil" }
        "2.000.000" -> cardinal { integer: "dos millones" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # Any single digit
        graph_digit = digit
        digits_no_one = (DAMO_DIGIT - "1") @ graph_digit

        # Any double digit
        graph_tens = teen
        graph_tens |= ties + (pynutil.delete("0") | (pynutil.insert(" y ") + graph_digit))
        graph_tens |= twenties

        self.tens = graph_tens.optimize()

        self.two_digit_non_zero = pynini.union(
            graph_digit, graph_tens, (pynini.cross("0", DAMO_SPACE) + graph_digit)
        ).optimize()

        # Three digit strings
        graph_hundreds = hundreds + pynini.union(
            pynutil.delete("00"),
            (insert_space + graph_tens),
            (pynini.cross("0", DAMO_SPACE) + graph_digit),
        )
        graph_hundreds |= pynini.cross("100", "cien")
        graph_hundreds |= (
            pynini.cross("1", "ciento")
            + insert_space
            + pynini.union(graph_tens, pynutil.delete("0") + graph_digit)
        )

        self.hundreds = graph_hundreds.optimize()

        # For all three digit strings with leading zeroes (graph appends '0's to manage place in string)
        graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + graph_tens)

        graph_hundreds_component_at_least_one_none_zero_digit = graph_hundreds_component | (
            pynutil.delete("00") + graph_digit
        )
        graph_hundreds_component_at_least_one_none_zero_digit_no_one = graph_hundreds_component | (
            pynutil.delete("00") + digits_no_one
        )

        graph_thousands_component_at_least_one_none_zero_digit = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_none_zero_digit,
            graph_hundreds_component_at_least_one_none_zero_digit_no_one
            + pynutil.insert(" mil")
            + (
                (insert_space + graph_hundreds_component_at_least_one_none_zero_digit)
                | pynutil.delete("000")
            ),
            pynini.cross("001", "mil")
            + (
                (insert_space + graph_hundreds_component_at_least_one_none_zero_digit)
                | pynutil.delete("000")
            ),
        )

        graph_thousands_component_at_least_one_none_zero_digit_no_one = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_none_zero_digit_no_one,
            graph_hundreds_component_at_least_one_none_zero_digit_no_one
            + pynutil.insert(" mil")
            + (
                (insert_space + graph_hundreds_component_at_least_one_none_zero_digit)
                | pynutil.delete("000")
            ),
            pynini.cross("001", "mil")
            + (
                (insert_space + graph_hundreds_component_at_least_one_none_zero_digit)
                | pynutil.delete("000")
            ),
        )

        graph_million = pynutil.add_weight(pynini.cross("000001", "un millón"), -0.001)
        graph_million |= (
            graph_thousands_component_at_least_one_none_zero_digit_no_one
            + pynutil.insert(" millones")
        )
        graph_million |= pynutil.delete("000000")
        graph_million += insert_space

        graph_billion = pynutil.add_weight(pynini.cross("000001", "un billón"), -0.001)
        graph_billion |= (
            graph_thousands_component_at_least_one_none_zero_digit_no_one
            + pynutil.insert(" billones")
        )
        graph_billion |= pynutil.delete("000000")
        graph_billion += insert_space

        graph_trillion = pynutil.add_weight(pynini.cross("000001", "un trillón"), -0.001)
        graph_trillion |= (
            graph_thousands_component_at_least_one_none_zero_digit_no_one
            + pynutil.insert(" trillones")
        )
        graph_trillion |= pynutil.delete("000000")
        graph_trillion += insert_space

        graph = (
            graph_trillion
            + graph_billion
            + graph_million
            + (graph_thousands_component_at_least_one_none_zero_digit | pynutil.delete("000000"))
        )

        self.graph = (
            ((DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT, 0))
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", DAMO_SIGMA)
            @ DAMO_DIGIT**24
            @ graph
            @ pynini.cdrewrite(delete_space, "[BOS]", "", DAMO_SIGMA)
            @ pynini.cdrewrite(delete_space, "", "[EOS]", DAMO_SIGMA)
            @ pynini.cdrewrite(
                pynini.cross(pynini.closure(DAMO_WHITE_SPACE, 2), DAMO_SPACE),
                DAMO_ALPHA,
                DAMO_ALPHA,
                DAMO_SIGMA,
            )
        )
        self.graph |= zero

        self.graph = filter_punctuation(self.graph).optimize()

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1
        )

        final_graph = (
            optional_minus_graph + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
