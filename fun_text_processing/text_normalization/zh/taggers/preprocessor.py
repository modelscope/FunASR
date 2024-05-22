import pynini
from fun_text_processing.text_normalization.zh.graph_utils import FUN_SIGMA, GraphFst
from fun_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class PreProcessor(GraphFst):
    """
    Preprocessing of TN:
        1. interjections removal such as '啊, 呃'
        2. fullwidth -> halfwidth char conversion
    好啊 -> 好
    呃对 -> 对
    ：   -> :
    ；   -> ;
    """

    def __init__(
        self,
        remove_interjections: bool = True,
        fullwidth_to_halfwidth: bool = True,
    ):
        super().__init__(name="PreProcessor", kind="processor")

        graph = pynini.cdrewrite("", "", "", FUN_SIGMA)

        if remove_interjections:
            remove_interjections_graph = pynutil.delete(
                pynini.string_file(get_abs_path("data/denylist/denylist.tsv"))
            )
            graph @= pynini.cdrewrite(remove_interjections_graph, "", "", FUN_SIGMA)

        if fullwidth_to_halfwidth:
            fullwidth_to_halfwidth_graph = pynini.string_file(
                get_abs_path("data/char/fullwidth_to_halfwidth.tsv")
            )
            graph @= pynini.cdrewrite(fullwidth_to_halfwidth_graph, "", "", FUN_SIGMA)

        self.fst = graph.optimize()
