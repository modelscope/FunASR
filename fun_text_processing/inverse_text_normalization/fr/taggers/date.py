import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    GraphFst,
    delete_extra_space,
)
from fun_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, in the form of (day) month (year) or year
        e.g. le vingt-quatre juillet deux-mille-treize -> date { day: "24" month: "juli" year: "2013" preserve_order: true }
        e.g. le vingt-quatre juillet deux-mille-treize -> date { day: "24" month: "juli" year: "2013" preserve_order: true }
        e.g. le premier janvier -> date { day: "1" month: "janvier"  preserve_order: true }

    Also will convert colloquialism of spelling in which tens of hundreds are used to express date. (e.g. nineteen hundred and four)
        e.g. le vingt mais dix-neuf-cent-quatre -> date { day: "20" month: "mais" year: "1904" preserve_order: true }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        self.cardinal = cardinal.graph_no_exception

        year_graph = self.cardinal

        month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
        month_graph = pynutil.insert('month: "') + month_graph + pynutil.insert('"')

        day_graph = self.cardinal | pynini.cross(
            "premier", "1"
        )  # Premier is only ordinal used for dates
        day_graph = pynutil.insert('day: "') + day_graph + pynutil.insert('"')
        optional_graph_year = pynini.closure(
            delete_extra_space + pynutil.insert('year: "') + year_graph + pynutil.insert('"'),
            0,
            1,
        )
        graph_dmy = day_graph + delete_extra_space + month_graph + optional_graph_year

        final_graph = graph_dmy
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
