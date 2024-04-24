import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { day: "1" month: "janvier" preserve_order: true } -> 1 de enero
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")

        convert_primer = pynini.cross("1", "1ᵉʳ")
        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete('"')
            + (
                pynini.closure(DAMO_NOT_QUOTE, 1) | pynutil.add_weight(convert_primer, -1)
            )  # first of the month is ordinal
            + pynutil.delete('"')
        )
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # day month
        graph_dm = day + delete_extra_space + month
        graph_dmy = graph_dm + delete_extra_space + year

        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete("field_order:")
            + delete_space
            + pynutil.delete('"')
            + DAMO_NOT_QUOTE
            + pynutil.delete('"')
            + delete_space
        )

        final_graph = (graph_dm | graph_dmy) + delete_space + optional_preserve_order

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
