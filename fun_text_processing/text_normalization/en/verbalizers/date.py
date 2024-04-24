import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.examples import plurals
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { month: "february" day: "five" year: "twenty twelve" preserve_order: true } -> february fifth twenty twelve
        date { day: "five" month: "february" year: "twenty twelve" preserve_order: true } -> the fifth of february twenty twelve

    Args:
        ordinal: OrdinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, ordinal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        month = pynini.closure(DAMO_NOT_QUOTE, 1)
        day_cardinal = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        day = day_cardinal @ ordinal.suffix

        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete('"')
            + month
            + pynutil.delete('"')
        )

        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete('"')
        )

        # month (day) year
        graph_mdy = (
            month
            + pynini.closure(delete_extra_space + day, 0, 1)
            + pynini.closure(delete_extra_space + year, 0, 1)
        )
        # may 5 -> may five
        if not deterministic and not lm:
            graph_mdy |= (
                month
                + pynini.closure(delete_extra_space + day_cardinal, 0, 1)
                + pynini.closure(delete_extra_space + year, 0, 1)
            )

        # day month year
        graph_dmy = (
            pynutil.insert("the ")
            + day
            + delete_extra_space
            + pynutil.insert("of ")
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

        final_graph = (
            (
                plurals._priority_union(
                    graph_mdy, pynutil.add_weight(graph_dmy, 0.0001), DAMO_SIGMA
                )
                | year
            )
            + delete_space
            + optional_preserve_order
        )
        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
