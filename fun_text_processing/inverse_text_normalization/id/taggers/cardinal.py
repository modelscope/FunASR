import pynini
from fun_text_processing.inverse_text_normalization.id.utils import get_abs_path, num_to_word
from fun_text_processing.inverse_text_normalization.id.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus twenty three -> cardinal { integer: "23" negative: "-" } }
    Numbers below thirteen are not converted.
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_hundreds = pynini.string_file(get_abs_path("data/numbers/hundreds.tsv"))
        graph_thousand = pynini.string_file(get_abs_path("data/numbers/thousand.tsv"))

        graph_hundred = pynini.cross("ratus", "") | pynini.cross("seratus", "")

        graph_hundred_component = pynini.union(
            graph_digit + delete_space + graph_hundred, pynutil.insert("0")
        )
        graph_hundred_component += delete_space
        graph_hundred_component += pynini.union(
            graph_teen | pynutil.insert("00"),
            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
        )

        graph_one_hundred_component = pynini.union(
            pynini.cross("ratus", "1") | pynini.cross("seratus", "1")
        )
        graph_one_hundred_component += delete_space
        graph_one_hundred_component += pynini.union(
            graph_teen | pynutil.insert("00"),
            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
        )
        graph_hundred_component = graph_hundred_component | graph_one_hundred_component

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(DAMO_DIGIT) + (DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )
        graph_thousand = pynini.cross("ribu", "") | pynini.cross("seribu", "")
        graph_one_thousand_component = pynini.union(
            pynini.cross("ribu", "1") | pynini.cross("seribu", "1")
        )

        graph_thousands = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("ribu") | pynutil.delete("seribu")),
            pynutil.insert("000", weight=0.1),
        )
        graph_thousands = graph_thousands | (pynutil.insert("00") + graph_one_thousand_component)

        graph_million = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("juta") | pynutil.delete("sejuta")),
            pynutil.insert("000", weight=0.1),
        )
        graph_billion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (
                pynutil.delete("miliar")
                | pynutil.delete("semiliar")
                | pynutil.delete("milyar")
                | pynutil.delete("semilyar")
            ),
            pynutil.insert("000", weight=0.1),
        )
        graph_trillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("triliun") | pynutil.delete("setriliun")),
            pynutil.insert("000", weight=0.1),
        )
        graph_quadrillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("milion lipat empat"),
            pynutil.insert("000", weight=0.1),
        )
        graph_quintillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("triliun"),
            pynutil.insert("000", weight=0.1),
        )
        graph_sextillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("sextillion"),
            pynutil.insert("000", weight=0.1),
        )

        graph = pynini.union(
            graph_sextillion
            + delete_space
            + graph_quintillion
            + delete_space
            + graph_quadrillion
            + delete_space
            + graph_trillion
            + delete_space
            + graph_billion
            + delete_space
            + graph_million
            + delete_space
            + graph_thousands
            + delete_space
            + graph_hundred_component,
            # graph_digit,
            graph_zero,
        )

        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0"))
            + pynini.difference(DAMO_DIGIT, "0")
            + pynini.closure(DAMO_DIGIT),
            "0",
        )

        labels_exception = ["nol"]
        graph_exception = pynini.union(*labels_exception)

        graph = (
            pynini.cdrewrite(pynutil.delete("dan"), DAMO_SPACE, DAMO_SPACE, DAMO_SIGMA)
            @ (DAMO_ALPHA + DAMO_SIGMA)
            @ graph
        )

        self.graph_no_exception = graph

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("kurang", '"-"') + DAMO_SPACE, 0, 1
        )

        final_graph = (
            optional_minus_graph + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
