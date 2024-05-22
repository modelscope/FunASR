import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    GraphFst,
    insert_space,
)
from fun_text_processing.text_normalization.en.taggers.date import get_four_digit_year_graph
from fun_text_processing.text_normalization.en.utils import get_abs_path
from pynini.examples import plurals
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        -23 -> cardinal { negative: "true"  integer: "twenty three" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        self.lm = lm
        self.deterministic = deterministic
        # TODO replace to have "oh" as a default for "0"
        graph = pynini.Far(get_abs_path("data/number/cardinal_number_name.far")).get_fst()
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            pynini.closure(DAMO_DIGIT, 2, 3) | pynini.difference(DAMO_DIGIT, pynini.accep("0"))
        ) @ graph

        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))

        single_digits_graph = pynini.invert(graph_digit | graph_zero)
        self.single_digits_graph = single_digits_graph + pynini.closure(
            insert_space + single_digits_graph
        )

        if not deterministic:
            # for a single token allow only the same normalization
            # "007" -> {"oh oh seven", "zero zero seven"} not {"oh zero seven"}
            single_digits_graph_zero = pynini.invert(graph_digit | graph_zero)
            single_digits_graph_oh = pynini.invert(graph_digit) | pynini.cross("0", "oh")

            self.single_digits_graph = single_digits_graph_zero + pynini.closure(
                insert_space + single_digits_graph_zero
            )
            self.single_digits_graph |= single_digits_graph_oh + pynini.closure(
                insert_space + single_digits_graph_oh
            )

            single_digits_graph_with_commas = pynini.closure(
                self.single_digits_graph + insert_space, 1, 3
            ) + pynini.closure(
                pynutil.delete(",")
                + single_digits_graph
                + insert_space
                + single_digits_graph
                + insert_space
                + single_digits_graph,
                1,
            )

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1
        )

        graph = (
            pynini.closure(DAMO_DIGIT, 1, 3)
            + (pynini.closure(pynutil.delete(",") + DAMO_DIGIT**3) | pynini.closure(DAMO_DIGIT**3))
        ) @ graph

        self.graph = graph
        self.graph_with_and = self.add_optional_and(graph)

        if deterministic:
            long_numbers = pynini.compose(
                DAMO_DIGIT ** (5, ...), self.single_digits_graph
            ).optimize()
            final_graph = plurals._priority_union(
                long_numbers, self.graph_with_and, DAMO_SIGMA
            ).optimize()
            cardinal_with_leading_zeros = pynini.compose(
                pynini.accep("0") + pynini.closure(DAMO_DIGIT), self.single_digits_graph
            )
            final_graph |= cardinal_with_leading_zeros
        else:
            leading_zeros = pynini.compose(
                pynini.closure(pynini.accep("0"), 1), self.single_digits_graph
            )
            cardinal_with_leading_zeros = (
                leading_zeros
                + pynutil.insert(" ")
                + pynini.compose(pynini.closure(DAMO_DIGIT), self.graph_with_and)
            )

            # add small weight to non-default graphs to make sure the deterministic option is listed first
            final_graph = (
                self.graph_with_and
                | pynutil.add_weight(self.single_digits_graph, 0.0001)
                | get_four_digit_year_graph()  # allows e.g. 4567 be pronouced as forty five sixty seven
                | pynutil.add_weight(single_digits_graph_with_commas, 0.0001)
                | cardinal_with_leading_zeros
            )

        final_graph = (
            optional_minus_graph + pynutil.insert('integer: "') + final_graph + pynutil.insert('"')
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def add_optional_and(self, graph):
        graph_with_and = graph

        if not self.lm:
            graph_with_and = pynutil.add_weight(graph, 0.00001)
            not_quote = pynini.closure(DAMO_NOT_QUOTE)
            no_thousand_million = pynini.difference(
                not_quote, not_quote + pynini.union("thousand", "million") + not_quote
            ).optimize()
            integer = (
                not_quote
                + pynutil.add_weight(
                    pynini.cross("hundred ", "hundred and ") + no_thousand_million, -0.0001
                )
            ).optimize()

            no_hundred = pynini.difference(
                DAMO_SIGMA, not_quote + pynini.accep("hundred") + not_quote
            ).optimize()
            integer |= (
                not_quote
                + pynutil.add_weight(
                    pynini.cross("thousand ", "thousand and ") + no_hundred, -0.0001
                )
            ).optimize()

            optional_hundred = pynini.compose((DAMO_DIGIT - "0") ** 3, graph).optimize()
            optional_hundred = pynini.compose(
                optional_hundred, DAMO_SIGMA + pynini.cross(" hundred", "") + DAMO_SIGMA
            )
            graph_with_and |= pynini.compose(graph, integer).optimize()
            graph_with_and |= optional_hundred
        return graph_with_and
