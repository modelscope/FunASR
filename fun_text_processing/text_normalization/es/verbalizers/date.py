import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    delete_preserve_order,
)
from fun_text_processing.text_normalization.es.graph_utils import strip_cardinal_apocope
from fun_text_processing.text_normalization.es.taggers.date import articles
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { day: "treinta y uno" month: "marzo" year: "dos mil" } -> "treinta y uno de marzo de dos mil"
        date { day: "uno" month: "mayo" year: "del mil novecientos noventa" } -> "primero de mayo del mil novecientos noventa"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        day_cardinal = (
            pynutil.delete('day: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )
        day = strip_cardinal_apocope(day_cardinal)

        primero = pynini.cdrewrite(pynini.cross("uno", "primero"), "[BOS]", "[EOS]", DAMO_SIGMA)
        day = (
            (day @ primero) if deterministic else pynini.union(day, day @ primero)
        )  # Primero for first day is traditional, but will vary depending on region

        month = pynutil.delete('month: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')

        year = (
            pynutil.delete('year: "')
            + articles
            + DAMO_SPACE
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Insert preposition if wasn't originally with the year. This would mean a space was present
        year = pynutil.add_weight(year, -0.001)
        year |= (
            pynutil.delete('year: "')
            + pynutil.insert("de ")
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # day month year
        graph_dmy = (
            day
            + pynini.cross(DAMO_SPACE, " de ")
            + month
            + pynini.closure(pynini.accep(" ") + year, 0, 1)
        )

        graph_mdy = month + DAMO_SPACE + day + pynini.closure(DAMO_SPACE + year, 0, 1)
        if deterministic:
            graph_mdy += pynutil.delete(
                " preserve_order: true"
            )  # Only accepts this if was explicitly passed

        self.graph = graph_dmy | graph_mdy
        final_graph = self.graph + delete_preserve_order

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
