import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { day: "1" month: "enero" preserve_order: true } -> 1 de enero
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        day = (
            pynutil.delete("day:")
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
        graph_dmy = (
            day
            + delete_extra_space
            + pynutil.insert("de")
            + insert_space
            + month
            + (delete_extra_space + pynutil.insert("de") + insert_space + year).ques
        )
        graph_dmy |= (
            day
            + delete_space
            + pynutil.insert("/")
            + month
            + pynutil.delete(' morphosyntactic_features: "/"')
            + (delete_space + pynutil.insert("/") + year).ques
        )

        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete("field_order:")
            + delete_space
            + pynutil.delete('"')
            + DAMO_NOT_QUOTE
            + pynutil.delete('"')
            + delete_space
        )

        final_graph = graph_dmy + delete_space + optional_preserve_order

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
