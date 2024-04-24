import pynini
from fun_text_processing.inverse_text_normalization.pt.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import DAMO_SIGMA, GraphFst, delete_space
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        vigésimo primeiro -> ordinal { integer: "21" morphosyntactic_features: "o" }
    This class converts ordinal up to "milésimo" (one thousandth) exclusive.

    Cardinals below ten are not converted (in order to avoid
    e.g. "primero fez ..." -> "1º fez...", "segunda guerra mundial" -> "2ª guerra mundial"
    and any other odd conversions.)

    This FST also records the ending of the ordinal (called "morphosyntactic_features"):
    either "o" or "a".

    Args:
        cardinal: CardinalFst
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="classify")

        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/ordinals/ties.tsv"))
        graph_hundreds = pynini.string_file(get_abs_path("data/ordinals/hundreds.tsv"))

        ordinal_graph_union = pynini.union(
            pynutil.add_weight(graph_digit, 0.4),
            pynutil.add_weight(
                graph_ties + ((delete_space + graph_digit) | pynutil.insert("0")), 0.2
            ),
            graph_hundreds
            + ((delete_space + graph_ties) | pynutil.insert("0"))
            + ((delete_space + graph_digit) | pynutil.insert("0")),
        )

        accept_o_endings = DAMO_SIGMA + pynini.accep("o")
        accept_a_endings = DAMO_SIGMA + pynini.accep("a")

        ordinal_graph_o = accept_o_endings @ ordinal_graph_union
        ordinal_graph_a = accept_a_endings @ ordinal_graph_union

        # 'optional_numbers_in_front' have negative weight so we always
        # include them if they're there
        optional_in_front = (
            pynutil.add_weight(ordinal_graph_union, -0.1) + delete_space.closure()
        ).closure()
        graph_o_suffix = optional_in_front + ordinal_graph_o
        graph_a_suffix = optional_in_front + ordinal_graph_a

        # don't convert ordinals from one to nine inclusive
        graph_exception = pynini.project(pynini.union(graph_digit), "input")
        graph_o_suffix = (
            pynini.project(graph_o_suffix, "input") - graph_exception.arcsort()
        ) @ graph_o_suffix
        graph_a_suffix = (
            pynini.project(graph_a_suffix, "input") - graph_exception.arcsort()
        ) @ graph_a_suffix

        graph = (
            pynutil.insert('integer: "')
            + graph_o_suffix
            + pynutil.insert('"')
            + pynutil.insert(' morphosyntactic_features: "o"')
        )
        graph |= (
            pynutil.insert('integer: "')
            + graph_a_suffix
            + pynutil.insert('"')
            + pynutil.insert(' morphosyntactic_features: "a"')
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
