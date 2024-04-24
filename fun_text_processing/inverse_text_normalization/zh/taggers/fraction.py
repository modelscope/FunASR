from fun_text_processing.inverse_text_normalization.zh.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    convert_space,
    delete_space,
    delete_extra_space,
    DAMO_SIGMA,
    DAMO_CHAR,
    DAMO_SPACE,
)

import pynini
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")
        # integer_part # numerator # denominator
        graph_cardinal = cardinal.graph_no_exception

        # without the integerate part
        # 分子
        numerator = pynutil.insert('numerator: "') + graph_cardinal + pynutil.insert('"')
        # 分母
        denominator = (
            pynutil.insert('denominator: "')
            + graph_cardinal
            + pynutil.delete("分之")
            + pynutil.insert('"')
        )

        ##
        graph_fraction_component = denominator + pynutil.insert(" ") + numerator

        self.graph_fraction_component = graph_fraction_component

        graph = graph_fraction_component
        graph = graph.optimize()
        self.final_graph_wo_negative = graph

        ##负
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ")
            + (pynini.cross("负", '"true"') | pynini.cross("负的", '"true"'))
            + DAMO_SPACE,
            0,
            1,
        )

        graph = optional_graph_negative + graph
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
