import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    DAMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from fun_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "8" minutes: "30" suffix: "du matin"} -> 8 h 30
        time { hours: "8" minutes: "30" } -> 8 h 30
        time { hours: "8" minutes: "30" suffix: "du soir"} -> 20 h 30
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        hour_to_night = pynini.string_file(get_abs_path("data/time/hour_to_night.tsv"))

        day_suffixes = pynutil.delete('suffix: "am"')
        night_suffixes = pynutil.delete('suffix: "pm"')

        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_DIGIT, 1, 2)
            + pynutil.delete('"')
        )
        minute = (
            pynutil.delete("minutes:")
            + delete_extra_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_DIGIT, 1, 2)
            + pynutil.delete('"')
        )

        graph = (
            hour
            + delete_extra_space
            + pynutil.insert("h")
            + minute.ques
            + delete_space
            + day_suffixes.ques
        )

        graph |= (
            hour @ hour_to_night
            + delete_extra_space
            + pynutil.insert("h")
            + minute.ques
            + delete_space
            + night_suffixes
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
