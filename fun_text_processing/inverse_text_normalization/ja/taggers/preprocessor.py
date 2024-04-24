import pynini
from fun_text_processing.inverse_text_normalization.ja.graph_utils import DAMO_SIGMA, GraphFst
from fun_text_processing.inverse_text_normalization.ja.utils import get_abs_path
from pynini.lib import pynutil


class PreProcessor(GraphFst):
    def __init__(
        self,
        halfwidth_to_fullwidth: bool = True,
    ):
        super().__init__(name="PreProcessor", kind="processor")

        graph = pynini.cdrewrite("", "", "", DAMO_SIGMA)

        if halfwidth_to_fullwidth:
            halfwidth_to_fullwidth_graph = pynini.string_file(
                get_abs_path("data/char/halfwidth_to_fullwidth.tsv")
            )
            graph @= pynini.cdrewrite(halfwidth_to_fullwidth_graph, "", "", DAMO_SIGMA)

        self.fst = graph.optimize()
