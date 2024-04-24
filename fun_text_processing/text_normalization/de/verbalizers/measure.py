import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_preserve_order,
)
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, e.g.
        measure { cardinal { integer: "zwei" units: "unzen" } } -> "zwei unzen"
        measure { cardinal { integer_part: "zwei" quantity: "millionen" units: "unzen" } } -> "zwei millionen unzen"

    Args:
        decimal: decimal GraphFst
        cardinal: cardinal GraphFst
        fraction: fraction GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self, decimal: GraphFst, cardinal: GraphFst, fraction: GraphFst, deterministic: bool
    ):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)
        unit = pynutil.delete('units: "') + pynini.closure(DAMO_NOT_QUOTE) + pynutil.delete('"')

        graph_decimal = decimal.fst
        graph_cardinal = cardinal.fst
        graph_fraction = fraction.fst

        graph = (graph_cardinal | graph_decimal | graph_fraction) + pynini.accep(" ") + unit

        graph |= unit + delete_extra_space + (graph_cardinal | graph_decimal)
        graph += delete_preserve_order

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
