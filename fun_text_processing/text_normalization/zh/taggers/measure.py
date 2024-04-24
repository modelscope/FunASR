import pynini
from fun_text_processing.text_normalization.zh.graph_utils import GraphFst, insert_space
from fun_text_processing.text_normalization.zh.taggers.cardinal import Cardinal
from fun_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class Measure(GraphFst):
    """
    1kg  -> tokens { measure { cardinal { integer: "一" } units: "千克" } }
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        units_en = pynini.string_file(get_abs_path("data/measure/units_en.tsv"))
        units_zh = pynini.string_file(get_abs_path("data/measure/units_zh.tsv"))
        graph = (
            pynutil.insert("cardinal { ")
            + pynutil.insert('integer: "')
            + Cardinal().graph_cardinal
            + pynutil.insert('"')
            + pynutil.insert(" }")
            + insert_space
            + pynutil.insert('units: "')
            + (units_en | units_zh)
            + pynutil.insert('"')
        )
        percent_graph = (
            pynutil.insert("decimal { ")
            + pynutil.insert('integer_part: "')
            + Cardinal().graph_cardinal
            + pynutil.delete("%")
            + pynutil.insert('"')
            + pynutil.insert(" }")
        )
        graph |= percent_graph

        self.fst = self.add_tokens(graph).optimize()
