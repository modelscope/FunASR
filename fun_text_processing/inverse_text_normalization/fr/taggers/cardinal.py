import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    DAMO_CHAR,
    DAMO_DIGIT,
    DAMO_NOT_SPACE,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    delete_hyphen,
)
from fun_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


def rewrite(cardinal: "pynini.FstLike") -> "pynini.FstLike":
    """
    Function to rewrite cardinals written in traditional orthograph (no '-' for numbers >100)
    to current orthography ('-' between all words in number string)
    e.g. deux mille cent vingt-trois -> deux-mille-cent-vingt-trois.
    In cases where original orthography is current, or string is mixture of two orthographies,
    will render invalid form that will not pass through CardinalFst
    e.g. deux-mille cent-vingt-trois -> "deux##vingt-trois" ('#' is not accepted in cardinal FST and will fail to convert.)
    e.g. deux

    Args:
        cardinal: cardinal FST
    """

    # Traditional orthography does not hyphenate numbers > 100, this will insert hyphens in
    # those contexts.
    targets = pynini.string_map(
        [
            "et",  # for 'et un/onze'
            "cent",
            "mille",
            "million",
            "milliard",
            "billion",
            "billiard",
            "trillion",
            "trilliard",
        ]
    )
    targets += pynini.accep("s").ques

    no_spaces = pynini.closure(DAMO_NOT_SPACE)

    # Valid numbers in reformed orthography will have no spaces.
    new_orthography_sigma = no_spaces

    # Old orthography will not have these strings. Replacing with character to mark.
    targets_for_filtering = ("-" + targets) | ("-" + targets + "-") | (targets + "-")

    filter = pynini.cdrewrite(
        pynini.cross(targets_for_filtering, "#"), "", "", DAMO_SIGMA
    )  # Invalid for cardinal

    old_orthography_sigma = pynini.difference(
        DAMO_CHAR, "#"
    )  # Marked character removed from sigma_star.
    old_orthography_sigma.closure()

    # Only accept strings that occur in old orthography. (This avoids tying two non-related numbers together.)
    # e.g. mille cent-une -> mille-cent-une
    filter @= old_orthography_sigma

    # Now know replacements will only work around targets
    replace_left = pynini.cdrewrite(pynini.cross(" ", "-"), "", targets, DAMO_SIGMA)

    replace_right = pynini.cdrewrite(pynini.cross(" ", "-"), targets, "", DAMO_SIGMA)

    replace = replace_left @ replace_right

    graph = new_orthography_sigma | (filter @ replace)

    return graph @ cardinal


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. mois vingt-trois -> cardinal { negative: "-" integer: "23"}
    This class converts cardinals up to (but not including) "un-quatrillion",
    i.e up to "one septillion" in English (10^{24}).
    Cardinals below nine are not converted (in order to avoid
    "j'ai un pomme." --> "j'ai 1 pomme" and any other odd conversions.)
    This transducer accomodates both traditional hyphenation of numbers ('-' for most numbers <100)
    and current hyphenation (all elements of number are hyphenated), prioritizing the latter.
    e.g cent cinquante et un -> cardinal { integer: "151"}
        cent-cinquante-et-un -> cardinal { integer: "151"}
    This is done through a context dependent rewrite that attempts to map old spelling to new.
    e.g. cent cinquante et un -> cent-cinquante-et-un
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_teens = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_ties_unique = pynini.string_file(get_abs_path("data/numbers/ties_unique.tsv"))

        # Tens components
        graph_tens_component = graph_ties + ((delete_hyphen + graph_digit) | pynutil.insert("0"))
        graph_tens_component = pynini.union(graph_tens_component, graph_teens, graph_ties_unique)

        graph_tens_component_with_leading_zeros = pynini.union(
            graph_tens_component,
            (pynutil.insert("0") + (graph_digit | pynutil.insert("0", weight=0.01))),
        )

        # Hundreds components
        graph_cent_singular = pynutil.delete("cent")  # Used in hundreds place
        graph_cent_plural = pynini.cross(
            "cents", "00"
        )  # Only used as terminus of hundred sequence. deux cents -> 200, deux cent un -> 201

        graph_digit_no_one = pynini.project(pynini.union("un", "une"), "input")
        graph_digit_no_one = (
            pynini.project(graph_digit, "input") - graph_digit_no_one.arcsort()
        ) @ graph_digit

        graph_hundreds_component_singular = (
            graph_digit_no_one + delete_hyphen + graph_cent_singular
        )  # Regular way: [1-9] * 100

        graph_hundreds_component_singular = pynini.union(
            graph_hundreds_component_singular, pynini.cross("cent", "1")
        )
        graph_hundreds_component_singular += delete_hyphen
        graph_hundreds_component_singular += graph_tens_component_with_leading_zeros

        graph_hundreds_component_plural = graph_digit_no_one + delete_hyphen + graph_cent_plural

        graph_hundreds_component = pynini.union(
            graph_hundreds_component_singular,
            graph_hundreds_component_plural,
            pynutil.insert("0") + graph_tens_component_with_leading_zeros,
        )

        graph_hundreds_component_at_least_one_none_zero_digit = graph_hundreds_component @ (
            pynini.closure(DAMO_DIGIT) + (DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT)
        )
        self.graph_hundreds_component_at_least_one_none_zero_digit = rewrite(
            graph_hundreds_component_at_least_one_none_zero_digit
        ).optimize()

        # Graph thousands (we'll need this for cases of mille millions, mille milliards...)
        graph_tens_of_hundreds_component_singular = (
            graph_tens_component + delete_hyphen + graph_cent_singular
        )  # Tens of hundreds. e.g. 1900 = nineteen hundred/ 'dix neuf cents"
        graph_tens_of_hundreds_component_singular += (
            delete_hyphen + graph_tens_component_with_leading_zeros
        )
        graph_tens_of_hundreds_component_plural = (
            graph_tens_component + delete_hyphen + graph_cent_plural
        )
        graph_tens_of_hundred_component = (
            graph_tens_of_hundreds_component_plural | graph_tens_of_hundreds_component_singular
        )

        graph_thousands = pynini.union(
            graph_hundreds_component_at_least_one_none_zero_digit
            + delete_hyphen
            + pynutil.delete("mille"),
            pynutil.insert("001") + pynutil.delete("mille"),  # because 'mille', not 'un mille'
            pynutil.insert("000", weight=0.1),
        )

        # All other large amounts
        graph_millions = pynini.union(
            graph_hundreds_component_at_least_one_none_zero_digit
            + delete_hyphen
            + (pynutil.delete("million") | pynutil.delete("millions")),
            pynutil.insert("000", weight=0.1),
        )

        graph_milliards = pynini.union(  # French for English 'billion'
            graph_hundreds_component_at_least_one_none_zero_digit
            + delete_hyphen
            + (pynutil.delete("milliard") | pynutil.delete("milliards")),
            pynutil.insert("000", weight=0.1),
        )

        graph_billions = pynini.union(  # NOTE: this is English 'trillion.'
            graph_hundreds_component_at_least_one_none_zero_digit
            + delete_hyphen
            + (pynutil.delete("billions") | pynutil.delete("billion")),
            pynutil.insert("000", weight=0.1),
        )

        graph_mille_billion = pynini.union(
            graph_hundreds_component_at_least_one_none_zero_digit
            + delete_hyphen
            + pynutil.delete("mille"),
            pynutil.insert("001")
            + pynutil.delete("mille"),  # because we say 'mille', not 'un mille'
        )
        graph_mille_billion += delete_hyphen + (
            graph_millions | pynutil.insert("000") + pynutil.delete("billions")
        )  # allow for 'mil millones'
        graph_mille_billion |= pynutil.insert("000000", weight=0.1)

        graph_billiards = pynini.union(
            graph_hundreds_component_at_least_one_none_zero_digit
            + delete_hyphen
            + (pynutil.delete("billiards") | pynutil.delete("billiard")),
            pynutil.insert("000", weight=0.1),
        )

        graph_trillions = pynini.union(  # One thousand English trillions.
            graph_hundreds_component_at_least_one_none_zero_digit
            + delete_hyphen
            + (pynutil.delete("trillions") | pynutil.delete("trillion")),
            pynutil.insert("000", weight=0.1),
        )

        graph_trilliards = pynini.union(
            graph_hundreds_component_at_least_one_none_zero_digit
            + delete_hyphen
            + (pynutil.delete("trilliards") | pynutil.delete("trilliard")),
            pynutil.insert("000", weight=0.1),
        )

        graph = pynini.union(
            graph_trilliards
            + delete_hyphen
            + graph_trillions
            + delete_hyphen
            + graph_billiards
            + delete_hyphen
            + graph_billions
            + delete_hyphen
            + graph_milliards
            + delete_hyphen
            + graph_millions
            + delete_hyphen
            + graph_thousands
            + delete_hyphen
            + graph_hundreds_component,
            graph_tens_of_hundred_component,
            graph_zero,
        )

        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0"))
            + pynini.difference(DAMO_DIGIT, "0")
            + pynini.closure(DAMO_DIGIT),
            "0",
        )

        graph = rewrite(graph)

        self.graph_no_exception = graph.optimize()

        # save self.numbers_up_to_thousand for use in DecimalFst
        digits_up_to_thousand = DAMO_DIGIT | (DAMO_DIGIT**2) | (DAMO_DIGIT**3)
        numbers_up_to_thousand = pynini.compose(graph, digits_up_to_thousand).optimize()
        self.numbers_up_to_thousand = numbers_up_to_thousand

        # save self.numbers_up_to_million for use in DecimalFst
        digits_up_to_million = (
            DAMO_DIGIT
            | (DAMO_DIGIT**2)
            | (DAMO_DIGIT**3)
            | (DAMO_DIGIT**4)
            | (DAMO_DIGIT**5)
            | (DAMO_DIGIT**6)
        )
        numbers_up_to_million = pynini.compose(graph, digits_up_to_million).optimize()
        self.numbers_up_to_million = numbers_up_to_million

        # don't convert cardinals from zero to nine inclusive
        graph_exception = pynini.project(pynini.union(graph_digit, graph_zero), "input")

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("moins", '"-"') + DAMO_SPACE, 0, 1
        )

        final_graph = (
            optional_minus_graph + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
