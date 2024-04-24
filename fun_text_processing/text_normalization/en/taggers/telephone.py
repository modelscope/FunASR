import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_SIGMA,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
    plurals,
)
from fun_text_processing.text_normalization.en.utils import get_abs_path
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone, and IP, and SSN which includes country code, number part and extension
    country code optional: +***
    number part: ***-***-****, or (***) ***-****
    extension optional: 1-9999
    E.g
    +1 123-123-5678-1 -> telephone { country_code: "one" number_part: "one two three, one two three, five six seven eight" extension: "one" }
    1-800-GO-U-HAUL -> telephone { country_code: "one" number_part: "one, eight hundred GO U HAUL" }
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        add_separator = pynutil.insert(", ")  # between components
        zero = pynini.cross("0", "zero")
        if not deterministic:
            zero |= pynini.cross("0", pynini.union("o", "oh"))
        digit = (
            pynini.invert(pynini.string_file(get_abs_path("data/number/digit.tsv"))).optimize()
            | zero
        )

        telephone_prompts = pynini.string_file(get_abs_path("data/telephone/telephone_prompt.tsv"))
        country_code = (
            pynini.closure(telephone_prompts + delete_extra_space, 0, 1)
            + pynini.closure(pynini.cross("+", "plus "), 0, 1)
            + pynini.closure(digit + insert_space, 0, 2)
            + digit
            + pynutil.insert(",")
        )
        country_code |= telephone_prompts
        country_code = pynutil.insert('country_code: "') + country_code + pynutil.insert('"')
        country_code = (
            country_code + pynini.closure(pynutil.delete("-"), 0, 1) + delete_space + insert_space
        )

        area_part_default = pynini.closure(digit + insert_space, 2, 2) + digit
        area_part = pynini.cross("800", "eight hundred") | pynini.compose(
            pynini.difference(DAMO_SIGMA, "800"), area_part_default
        )

        area_part = (
            (area_part + (pynutil.delete("-") | pynutil.delete(".")))
            | (
                pynutil.delete("(")
                + area_part
                + (
                    (pynutil.delete(")") + pynini.closure(pynutil.delete(" "), 0, 1))
                    | pynutil.delete(")-")
                )
            )
        ) + add_separator

        del_separator = pynini.closure(pynini.union("-", " ", "."), 0, 1)
        number_length = ((DAMO_DIGIT + del_separator) | (DAMO_ALPHA + del_separator)) ** 7
        number_words = pynini.closure(
            (DAMO_DIGIT @ digit) + (insert_space | (pynini.cross("-", ", ")))
            | DAMO_ALPHA
            | (DAMO_ALPHA + pynini.cross("-", " "))
        )
        number_words |= pynini.closure(
            (DAMO_DIGIT @ digit) + (insert_space | (pynini.cross(".", ", ")))
            | DAMO_ALPHA
            | (DAMO_ALPHA + pynini.cross(".", " "))
        )
        number_words = pynini.compose(number_length, number_words)
        number_part = area_part + number_words
        number_part = pynutil.insert('number_part: "') + number_part + pynutil.insert('"')
        extension = (
            pynutil.insert('extension: "')
            + pynini.closure(digit + insert_space, 0, 3)
            + digit
            + pynutil.insert('"')
        )
        extension = pynini.closure(insert_space + extension, 0, 1)

        graph = plurals._priority_union(
            country_code + number_part, number_part, DAMO_SIGMA
        ).optimize()
        graph = plurals._priority_union(
            country_code + number_part + extension, graph, DAMO_SIGMA
        ).optimize()
        graph = plurals._priority_union(number_part + extension, graph, DAMO_SIGMA).optimize()

        # ip
        ip_prompts = pynini.string_file(get_abs_path("data/telephone/ip_prompt.tsv"))
        digit_to_str_graph = digit + pynini.closure(pynutil.insert(" ") + digit, 0, 2)
        ip_graph = digit_to_str_graph + (pynini.cross(".", " dot ") + digit_to_str_graph) ** 3
        graph |= (
            pynini.closure(
                pynutil.insert('country_code: "')
                + ip_prompts
                + pynutil.insert('"')
                + delete_extra_space,
                0,
                1,
            )
            + pynutil.insert('number_part: "')
            + ip_graph.optimize()
            + pynutil.insert('"')
        )
        # ssn
        ssn_prompts = pynini.string_file(get_abs_path("data/telephone/ssn_prompt.tsv"))
        three_digit_part = digit + (pynutil.insert(" ") + digit) ** 2
        two_digit_part = digit + pynutil.insert(" ") + digit
        four_digit_part = digit + (pynutil.insert(" ") + digit) ** 3
        ssn_separator = pynini.cross("-", ", ")
        ssn_graph = (
            three_digit_part + ssn_separator + two_digit_part + ssn_separator + four_digit_part
        )

        graph |= (
            pynini.closure(
                pynutil.insert('country_code: "')
                + ssn_prompts
                + pynutil.insert('"')
                + delete_extra_space,
                0,
                1,
            )
            + pynutil.insert('number_part: "')
            + ssn_graph.optimize()
            + pynutil.insert('"')
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
