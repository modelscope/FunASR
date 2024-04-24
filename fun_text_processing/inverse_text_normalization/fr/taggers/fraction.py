import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    DAMO_CHAR,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from fun_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        e.g. demi -> tokens { fraction { numerator: "1" denominator: "2" } }
        e.g. un et demi -> tokens { fraction { integer_part: "1" numerator: "1" denominator: "2" } }
        e.g. trois et deux centième -> tokens { fraction { integer_part: "3" numerator: "2" denominator: "100" } }

    Args:
        cardinal: OrdinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")
        # integer_part # numerator # denominator

        graph_cardinal = cardinal.graph_no_exception
        graph_strip_undo_root_change = pynini.string_file(
            get_abs_path("data/fractions.tsv")
        )  # add in absolute path

        graph_strip_no_root_change = pynutil.delete("ième")  # For no change to root
        graph_strip_no_root_change += pynutil.delete("s").ques  # for plurals

        graph_strip = graph_strip_no_root_change | graph_strip_undo_root_change

        self.fractional = ((pynini.closure(DAMO_CHAR) + graph_strip) @ graph_cardinal).optimize()

        integer = pynutil.insert('integer_part: "') + graph_cardinal + pynutil.insert('" ')
        integer += delete_space
        integer += pynutil.delete("et")  # used to demarcate integer and fractional parts

        numerator = pynutil.insert('numerator: "') + graph_cardinal + pynutil.insert('"')
        denominator = pynutil.insert(' denominator: "') + self.fractional + pynutil.insert('"')

        # Demi (half) can occur alone without explicit numerator.
        graph_demi_component = (
            pynutil.delete("demi") + pynutil.delete("e").ques + pynutil.delete("s").ques
        )
        graph_demi_component += pynutil.insert('numerator: "1" denominator: "2"')

        graph_fraction_component = numerator + delete_space + denominator
        graph_fraction_component |= graph_demi_component
        self.graph_fraction_component = graph_fraction_component

        graph = pynini.closure(integer + delete_space, 0, 1) + graph_fraction_component
        graph = graph.optimize()
        self.final_graph_wo_negative = graph

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("moins", '"true"') + delete_extra_space,
            0,
            1,
        )
        graph = optional_graph_negative + graph
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
