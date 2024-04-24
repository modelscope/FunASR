import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
)
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "true" integer_part: "12"  fractional_part: "5006" quantity: "billion" } -> -12.5006 billion

    Args:
        tn_decimal_verbalizer: TN decimal verbalizer
    """

    def __init__(self, tn_decimal_verbalizer: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)
        delete_space = pynutil.delete(" ")
        optional_sign = pynini.closure(
            pynutil.delete('negative: "') + DAMO_NOT_QUOTE + pynutil.delete('"') + delete_space,
            0,
            1,
        )
        optional_integer = pynini.closure(tn_decimal_verbalizer.integer, 0, 1)
        optional_fractional = pynini.closure(
            delete_space + pynutil.insert(",") + tn_decimal_verbalizer.fractional_default, 0, 1
        )
        graph = (
            optional_integer + optional_fractional + tn_decimal_verbalizer.optional_quantity
        ).optimize()
        self.numbers = optional_sign + graph
        graph = self.numbers + delete_preserve_order
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
