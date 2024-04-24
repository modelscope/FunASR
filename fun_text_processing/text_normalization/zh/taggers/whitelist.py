import pynini
from fun_text_processing.text_normalization.zh.graph_utils import GraphFst
from fun_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class Whitelist(GraphFst):
    """
    ATM  -> tokens { whitelist: "ATM" }
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        whitelist = pynini.string_file(get_abs_path("data/whitelist/default.tsv"))
        erhua = pynutil.insert('erhua: "') + pynini.accep("儿") + pynutil.insert('"')
        sign = pynini.string_file(get_abs_path("data/math/symbol.tsv"))
        whitelist = (
            pynutil.insert('name: "')
            + (pynini.string_file(get_abs_path("data/erhua/whitelist.tsv")) | whitelist | sign)
            + pynutil.insert('"')
        )
        graph = pynutil.add_weight(erhua, 0.1) | whitelist

        self.fst = graph.optimize()
