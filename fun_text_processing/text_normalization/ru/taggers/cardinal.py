# Copyright 2017 Google Inc.


# Adapted from https://github.com/google/TextNormalizationCoveringGrammars
# Russian minimally supervised number grammar.

import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    insert_space,
)
from fun_text_processing.text_normalization.ru.alphabet import RU_ALPHA, TO_CYRILLIC
from fun_text_processing.text_normalization.ru.utils import get_abs_path
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "1 001" ->  cardinal { integer: "тысяча один" }

    Args:
        number_names: number_names for cardinal and ordinal numbers
        alternative_formats: alternative number formats
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, number_names: dict, alternative_formats: dict, deterministic: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        self.cardinal_numbers_default = self.get_cardinal_numbers(
            number_names, alternative_formats, mode="all"
        )
        self.cardinal_numbers_nominative = self.get_cardinal_numbers(
            number_names, alternative_formats, mode="nominative"
        )
        self.optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true"') + insert_space, 0, 1
        )

        self.cardinal_numbers_with_optional_negative = (
            self.optional_graph_negative
            + pynutil.insert('integer: "')
            + self.cardinal_numbers_default
            + pynutil.insert('"')
        )

        # "03" -> remove leading zeros and verbalize
        leading_zeros = pynini.closure(pynini.cross("0", ""))
        self.cardinal_numbers_with_leading_zeros = (
            leading_zeros + self.cardinal_numbers_default
        ).optimize()

        # "123" -> "один два три"
        single_digits_graph = pynini.string_file(
            get_abs_path("data/numbers/cardinals_nominative_case.tsv")
        ).optimize()
        single_digits_graph = pynini.compose(DAMO_DIGIT, single_digits_graph)
        self.single_digits_graph = single_digits_graph + pynini.closure(
            insert_space + single_digits_graph
        )

        optional_quantity = pynini.string_file(get_abs_path("data/numbers/quantity.tsv")).optimize()
        optional_quantity = pynutil.insert('quantity: "') + optional_quantity + pynutil.insert('"')
        optional_quantity = pynini.closure(
            (pynutil.add_weight(pynini.accep(DAMO_SPACE), -0.1) | insert_space) + optional_quantity,
            0,
            1,
        )

        serial_graph = self.get_serial_graph()

        final_graph = (
            self.optional_graph_negative
            + pynutil.insert('integer: "')
            + self.cardinal_numbers_with_leading_zeros
            + pynutil.insert('"')
            + optional_quantity
        ).optimize()

        final_graph = pynutil.add_weight(final_graph, -0.1)
        final_graph |= (
            pynutil.insert('integer: "')
            + pynutil.add_weight(serial_graph, 10)
            + pynutil.insert('"')
        )
        self.final_graph = final_graph

        # to cover cases "2-х" -> "двух" (this is not covered by ordinal endings)
        final_graph |= pynini.compose(
            pynini.compose(DAMO_DIGIT ** (1, ...) + pynini.cross("-х", ""), final_graph),
            DAMO_SIGMA + pynini.accep('х"') + DAMO_SIGMA,
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def get_cardinal_numbers(
        self, number_names: dict, alternative_formats: dict, mode: str = "all"
    ):
        """Returns cardinal numbers names graph.

        Args:
            number_names: number_names for cardinal and ordinal numbers
            alternative_formats: alternative number formats
            mode: "all" - to return graph that includes all Ru cases, "nominative" to return only the nominative form
        """
        if mode == "all":
            cardinal_names = number_names["cardinal_number_names"]
        elif mode == "nominative":
            cardinal_names = number_names["cardinal_names_nominative"]
        else:
            raise ValueError(f"{mode} is not supported.")
        one_thousand_alternative = alternative_formats["one_thousand_alternative"]
        separators = alternative_formats["separators"]

        cardinal_numbers = cardinal_names | pynini.compose(cardinal_names, one_thousand_alternative)
        cardinal_numbers = pynini.compose(separators, cardinal_numbers)
        return cardinal_numbers

    def get_serial_graph(self):
        """
        Finite state transducer for classifying serial.
            The serial is a combination of digits, letters and dashes, e.g.:
            c325-b -> tokens { cardinal { integer: "си три два пять би" } }
        """
        num_graph = self.single_digits_graph

        alpha = TO_CYRILLIC | RU_ALPHA

        delimiter = insert_space | pynini.cross("-", " ") | pynini.cross("/", " ")
        letter_num = pynini.closure(alpha + delimiter, 1) + num_graph
        num_letter = pynini.closure(num_graph + delimiter, 1) + alpha
        num_delimiter_num = pynini.closure(num_graph + delimiter, 1) + num_graph
        next_alpha_or_num = pynini.closure(delimiter + (alpha | num_graph))
        serial_graph = (letter_num | num_letter | num_delimiter_num) + next_alpha_or_num

        # at least 1 alpha and 1 digit is present
        at_least_one_alpha_num = (
            DAMO_SIGMA
            + (RU_ALPHA | pynini.project(TO_CYRILLIC, "input"))
            + DAMO_SIGMA
            + DAMO_DIGIT
            + DAMO_SIGMA
        ) | (
            DAMO_SIGMA
            + DAMO_DIGIT
            + DAMO_SIGMA
            + (RU_ALPHA | pynini.project(TO_CYRILLIC, "input"))
            + DAMO_SIGMA
        )
        serial_graph = pynini.compose(at_least_one_alpha_num, serial_graph.optimize()).optimize()
        # numbers only with 2+ delimiters
        serial_graph |= (
            num_graph
            + delimiter
            + num_graph
            + delimiter
            + num_graph
            + pynini.closure(delimiter + num_graph)
        ).optimize()
        return serial_graph.optimize()
