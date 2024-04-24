import pynini
from fun_text_processing.inverse_text_normalization.es.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import DAMO_SIGMA, GraphFst, delete_space
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        vigésimo primero -> ordinal { integer: "21" morphosyntactic_features: "o" }
    This class converts ordinal up to "millesímo" (one thousandth) exclusive.

    Cardinals below ten are not converted (in order to avoid
    e.g. "primero hice ..." -> "1.º hice...", "segunda guerra mundial" -> "2.ª guerra mundial"
    and any other odd conversions.)

    This FST also records the ending of the ordinal (called "morphosyntactic_features"):
    either "o", "a", or "er".

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_teens = pynini.string_file(get_abs_path("data/ordinals/teen.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("data/ordinals/twenties.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/ordinals/ties.tsv"))
        graph_hundreds = pynini.string_file(get_abs_path("data/ordinals/hundreds.tsv"))

        ordinal_graph_union = pynini.union(
            graph_digit,
            graph_teens,
            graph_twenties,
            graph_ties,
            graph_hundreds,
        )

        accept_o_endings = DAMO_SIGMA + pynini.accep("o")
        accept_a_endings = DAMO_SIGMA + pynini.accep("a")
        accept_er_endings = DAMO_SIGMA.closure() + pynini.accep("er")

        ordinal_graph_o = accept_o_endings @ ordinal_graph_union
        ordinal_graph_a = accept_a_endings @ ordinal_graph_union
        ordinal_graph_er = accept_er_endings @ ordinal_graph_union

        # 'optional_numbers_in_front' have negative weight so we always
        # include them if they're there
        optional_numbers_in_front = (
            pynutil.add_weight(ordinal_graph_union, -0.1) + delete_space.closure()
        ).closure()
        graph_o_suffix = (optional_numbers_in_front + ordinal_graph_o) @ cardinal_graph
        graph_a_suffix = (optional_numbers_in_front + ordinal_graph_a) @ cardinal_graph
        graph_er_suffix = (optional_numbers_in_front + ordinal_graph_er) @ cardinal_graph

        # don't convert ordinals from one to nine inclusive
        graph_exception = pynini.project(pynini.union(graph_digit), "input")
        graph_o_suffix = (
            pynini.project(graph_o_suffix, "input") - graph_exception.arcsort()
        ) @ graph_o_suffix
        graph_a_suffix = (
            pynini.project(graph_a_suffix, "input") - graph_exception.arcsort()
        ) @ graph_a_suffix
        graph_er_suffix = (
            pynini.project(graph_er_suffix, "input") - graph_exception.arcsort()
        ) @ graph_er_suffix

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
        graph |= (
            pynutil.insert('integer: "')
            + graph_er_suffix
            + pynutil.insert('"')
            + pynutil.insert(' morphosyntactic_features: "er"')
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
