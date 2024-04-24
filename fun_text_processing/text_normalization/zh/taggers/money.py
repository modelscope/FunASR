import pynini
from fun_text_processing.text_normalization.zh.graph_utils import GraphFst, insert_space
from fun_text_processing.text_normalization.zh.taggers.cardinal import Cardinal
from fun_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class Money(GraphFst):
    """
    ￥1.25 -> tokens { money { fractional_part: "元" integer_part: "一点五" } }
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        currency_code = pynini.string_file(get_abs_path("data/money/currency_code.tsv"))
        currency_symbol = pynini.string_file(get_abs_path("data/money/currency_symbol.tsv"))
        graph = (
            pynutil.insert('fractional_part: "')
            + (currency_code | currency_symbol)
            + pynutil.insert('"')
            + insert_space
            + pynutil.insert('integer_part: "')
            + Cardinal().graph_cardinal
            + pynutil.insert('"')
        )

        self.fst = self.add_tokens(graph).optimize()
