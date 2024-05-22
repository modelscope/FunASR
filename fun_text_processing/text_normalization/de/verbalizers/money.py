import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
)
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { currency_maj: "euro" integer_part: "ein"} -> "ein euro"
        money { currency_maj: "euro" integer_part: "eins" fractional_part: "null null eins"} -> "eins komma null null eins euro"
        money { integer_part: "ein" currency_maj: "pfund" fractional_part: "vierzig" preserve_order: true} -> "ein pfund vierzig"
        money { integer_part: "ein" currency_maj: "pfund" fractional_part: "vierzig" currency_min: "pence" preserve_order: true} -> "ein pfund vierzig pence"
        money { fractional_part: "ein" currency_min: "penny" preserve_order: true} -> "ein penny"
        money { currency_maj: "pfund" integer_part: "null" fractional_part: "null eins" quantity: "million"} -> "null komma null eins million pfund"

    Args:
        decimal: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        keep_space = pynini.accep(" ")

        maj = (
            pynutil.delete('currency_maj: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        min = (
            pynutil.delete('currency_min: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        fractional_part = (
            pynutil.delete('fractional_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        integer_part = (
            pynutil.delete('integer_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_add_and = pynini.closure(pynutil.insert("und "), 0, 1)

        #  *** currency_maj
        graph_integer = integer_part + keep_space + maj

        #  *** currency_maj + (***) | ((und) *** current_min)
        graph_integer_with_minor = (
            integer_part
            + keep_space
            + maj
            + keep_space
            + (fractional_part | (optional_add_and + fractional_part + keep_space + min))
            + delete_preserve_order
        )

        # *** komma *** currency_maj
        graph_decimal = decimal.fst + keep_space + maj

        # *** current_min
        graph_minor = fractional_part + keep_space + min + delete_preserve_order

        graph = graph_integer | graph_integer_with_minor | graph_decimal | graph_minor

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
