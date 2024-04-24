import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    DAMO_SIGMA,
    GraphFst,
    delete_space,
)
from fun_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        vingt-deuxième -> ordinal { integer: "22" morphosyntactic_features: "e" }

    Also notes specific nouns that have unique normalization conventions.
    For instance, 'siècles' are rendered in roman numerals when given an ordinal adjective.
    e.g. dix-neuvième siècle -> XIXe

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        graph_cardinal = cardinal.graph_no_exception
        graph_undo_root_change = pynini.string_file(
            get_abs_path("data/ordinals/digits_root_change.tsv")
        )  # Returns base number to normal after root change.
        graph_firsts = pynini.string_file(get_abs_path("data/ordinals/firsts.tsv"))
        graph_second = pynini.string_file(get_abs_path("data/ordinals/second.tsv"))
        graph_special_ordinals = pynini.string_file(get_abs_path("data/ordinals/key_nouns.tsv"))

        # Removes morpheme
        graph_no_root_change = pynutil.delete("ième")  # For no change to root

        graph_strip_morpheme = pynini.union(graph_no_root_change, graph_undo_root_change)
        graph_strip_morpheme = DAMO_SIGMA + graph_strip_morpheme

        graph_integer_component = graph_strip_morpheme @ graph_cardinal

        graph_morpheme_component = pynutil.insert("e")  # Put the superscript in.
        graph_morpheme_component += pynini.accep("s").ques  # In case of plurals.

        # Concatenate with cardinal graph.
        graph_ordinal = pynutil.insert('integer: "') + graph_integer_component + pynutil.insert('"')
        graph_ordinal += (
            pynutil.insert(' morphosyntactic_features: "') + graph_morpheme_component
        )  # Leave open in case further morphems occur

        # Primer has a different subscript depending on gender, need to take note if
        # 'premier' or 'première'
        graph_firsts = pynutil.insert('integer: "1" morphosyntactic_features: "') + graph_firsts

        # Second used 'd' as a superscript.
        graph_second = pynutil.insert('integer: "2" morphosyntactic_features: "') + graph_second

        graph = graph_firsts | graph_second | graph_ordinal

        # For roman numerals. Carries over designation to verbalizer
        graph_special_ordinals = pynutil.insert("/") + delete_space + graph_special_ordinals

        graph += graph_special_ordinals.ques + pynutil.insert('"')

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
