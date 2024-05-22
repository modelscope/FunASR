#!/usr/bin/python
# -*- coding: utf-8 -*-

import pynini
from pynini import accep, cross, string_file, union
from pynini.lib.pynutil import delete, insert, add_weight
from fun_text_processing.inverse_text_normalization.ja.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.ja.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_CHAR,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil
import unicodedata


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus twenty three -> cardinal { integer: "23" negative: "-" } }
    Numbers below thirteen are not converted.
    """

    def __init__(self, enable_standalone_number: bool = True, enable_0_to_9: bool = True):
        super().__init__(name="cardinal", kind="classify")
        self.enable_standalone_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9
        zero = string_file(get_abs_path("data/numbers/zero.tsv"))
        digit = string_file(get_abs_path("data/numbers/digit.tsv"))
        hundred_digit = string_file(get_abs_path("data/numbers/hundred_digit.tsv"))
        sign = string_file(get_abs_path("data/numbers/sign.tsv"))
        dot = string_file(get_abs_path("data/numbers/dot.tsv"))
        ties = string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = string_file(get_abs_path("data/numbers/teen.tsv"))

        addzero = insert("0")
        digits = zero | digit  # 0 ~ 9
        teen = graph_teen
        teen |= cross("十", "1") + (digit | addzero)
        tens = ties + addzero | (ties + (digit | addzero))

        hundred = (
            digit
            + delete("百")
            + (
                tens
                | teen
                | add_weight(zero + digit, 0.1)
                | add_weight(digit + addzero, 0.5)
                | add_weight(addzero**2, 1.0)
            )
        )
        hundred |= cross("百", "1") + (
            tens
            | teen
            | add_weight(zero + digit, 0.1)
            | add_weight(digit + addzero, 0.5)
            | add_weight(addzero**2, 1.0)
        )
        hundred |= hundred_digit

        thousand = (
            (hundred | teen | tens | digits)
            + delete("千")
            + (
                hundred
                | add_weight(zero + tens, 0.1)
                | add_weight(addzero + zero + digit, 0.5)
                | add_weight(digit + addzero**2, 0.8)
                | add_weight(addzero**3, 1.0)
            )
        )
        ten_thousand = (
            (thousand | hundred | teen | tens | digits)
            + delete("万")
            + (
                thousand
                | add_weight(zero + hundred, 0.1)
                | add_weight(addzero + zero + tens, 0.5)
                | add_weight(addzero + addzero + zero + digit, 0.5)
                | add_weight(digit + addzero**3, 0.8)
                | add_weight(addzero**4, 1.0)
            )
        )

        hundred_thousand = (
            (ten_thousand | thousand | hundred | teen | tens | digits)
            + delete("十万")
            + (
                ten_thousand
                | add_weight(zero + thousand, 0.1)
                | add_weight(addzero + zero + hundred, 0.5)
                | add_weight(addzero + addzero + zero + tens, 0.5)
                | add_weight(addzero**3 + zero + digit, 0.5)
                | add_weight(digit + addzero**4, 0.8)
                | add_weight(addzero**5, 1.0)
            )
        )

        million = (
            (hundred_thousand | ten_thousand | thousand | hundred | teen | tens | digits)
            + delete("百万")
            + (
                hundred_thousand
                | add_weight(zero + ten_thousand, 0.1)
                | add_weight(addzero + zero + thousand, 0.5)
                | add_weight(addzero + addzero + zero + hundred, 0.5)
                | add_weight(addzero**3 + zero + tens, 0.5)
                | add_weight(addzero**4 + zero + digit, 0.5)
                | add_weight(digit + addzero**5, 0.8)
                | add_weight(addzero**6, 1.0)
            )
        )
        # 1亿
        hundred_million = (
            (million | hundred_thousand | ten_thousand | thousand | hundred | teen | tens | digits)
            + delete("億")
            + (
                add_weight(zero + million, 0.1)
                | add_weight(addzero + zero + hundred_thousand, 0.5)
                | add_weight(addzero**2 + zero + ten_thousand, 0.5)
                | add_weight(addzero**3 + zero + thousand, 0.5)
                | add_weight(addzero**4 + hundred, 0.5)
                | add_weight(addzero**5 + tens, 0.5)
                | add_weight(addzero**6 + digit, 0.5)
                | add_weight(digit + addzero**7, 0.8)
                | add_weight(addzero**8, 1.0)
            )
        )
        # 1兆
        hundred_billion = (
            (
                hundred_million
                | million
                | hundred_thousand
                | ten_thousand
                | thousand
                | hundred
                | teen
                | tens
                | digits
            )
            + delete("兆")
            + (
                add_weight(addzero**3 + zero + hundred_million, 0.1)
                | add_weight(addzero**4 + zero + million, 0.5)
                | add_weight(addzero**5 + zero + hundred_thousand, 0.5)
                | add_weight(addzero**6 + zero + ten_thousand, 0.5)
                | add_weight(addzero**7 + zero + thousand, 0.5)
                | add_weight(addzero**8 + hundred, 0.5)
                | add_weight(addzero**9 + tens, 0.5)
                | add_weight(addzero**10 + digit, 0.5)
                | add_weight(digit + addzero**11, 0.8)
                | add_weight(addzero**12, 1.0)
            )
        )
        # 1.11, 1.01
        number = (
            digits | teen | tens | hundred | thousand | ten_thousand | hundred_thousand | million
        )
        # number = digits | teen | tens | hundred | thousand | ten_thousand | hundred_thousand | million | hundred_million | hundred_billion
        # 兆/亿
        number = (number + accep("兆") + delete("零").ques).ques + (
            number + accep("億") + delete("零").ques
        ).ques + number | (number + accep("兆") + delete("〇").ques).ques + (
            number + accep("億") + delete("〇").ques
        ).ques + number

        number = sign.ques + number + (dot + digits.plus).ques
        self.number = number.optimize()
        self.digits = digits.optimize()

        # cardinal string like 127.0.0.1, used in ID, IP, etc.
        cardinal = digit.plus + (dot + digits.plus).plus
        # float number like 1.11
        cardinal |= number + dot + digits.plus
        # cardinal string like 110 or 12306 or 13125617878, used in phone
        cardinal |= digits**3 | digits**5 | digits**10 | digits**11 | digits**12
        # cardinal string like 23
        if self.enable_standalone_number:
            if self.enable_0_to_9:
                cardinal |= number
            else:
                number_two_plus = (
                    (digits + digits.plus)
                    | teen
                    | tens
                    | hundred
                    | thousand
                    | ten_thousand
                    | hundred_thousand
                    | million
                    | hundred_million
                    | hundred_billion
                )
                cardinal |= number_two_plus
        labels_exception = [""]
        graph_exception = pynini.union(*labels_exception)

        self.graph_no_exception = cardinal
        self.graph = (pynini.project(cardinal, "input") - graph_exception.arcsort()) @ cardinal

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("マイナス", '"-"') + DAMO_SPACE, 0, 1
        )

        final_graph = (
            optional_minus_graph + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

        # ########
        graph_hundred = pynini.cross("百", "")
        graph_a_hundred_digit_component = pynini.union(pynini.cross("百", "10") + digit)
        graph_one_hundred_component = pynini.union(pynini.cross("百", "100"))
        graph_hundred_ties_component = pynini.cross("百", "1") + pynini.union(
            graph_teen | pynutil.insert("00"),
            (ties | pynutil.insert("0")) + (digit | pynutil.insert("0")),
        )
        graph_hundred_component = pynini.union(digit + graph_hundred, pynutil.insert("0"))
        graph_hundred_component += pynini.union(
            graph_teen | pynutil.insert("00"),
            (ties | pynutil.insert("0")) + (digit | pynutil.insert("0")),
        )
        graph_hundred_component = (
            graph_hundred_component
            | graph_a_hundred_digit_component
            | graph_one_hundred_component
            | graph_hundred_ties_component
        )
        #
        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(DAMO_DIGIT) + (DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )
