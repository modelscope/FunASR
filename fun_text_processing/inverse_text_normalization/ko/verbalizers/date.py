import pynini
from fun_text_processing.inverse_text_normalization.ko.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { month: "january" day: "5" year: "2012" preserve_order: true } -> february 5 2012
        date { day: "5" month: "january" year: "2012" preserve_order: true } -> 5 february 2012
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + pynutil.insert(" ")
        )
        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + pynutil.insert(" ")
        )
        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + pynutil.insert(" ")
        )

        # month (day) year
        graph_mdy = (
            month
            + pynini.closure(delete_extra_space + day, 0, 1)
            + pynini.closure(delete_extra_space + year, 0, 1)
        )

        # (day) month year
        graph_dmy = (
            pynini.closure(day + delete_extra_space, 0, 1)
            + month
            + pynini.closure(delete_extra_space + year, 0, 1)
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

        # year month day
        graph_ymd = year + month + day

        # month day
        graph_md = month + day

        # year month
        graph_ym = year + month

        # add some grammars
        final_graph = (
            (graph_mdy | year | graph_dmy | graph_ymd | graph_md | graph_ym | month | day)
            + delete_space
            + optional_preserve_order
        )

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
