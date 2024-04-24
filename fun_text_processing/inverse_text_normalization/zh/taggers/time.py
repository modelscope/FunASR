import pynini
from fun_text_processing.inverse_text_normalization.zh.taggers.cardinal import CardinalFst
from fun_text_processing.inverse_text_normalization.zh.utils import get_abs_path, num_to_word
from fun_text_processing.inverse_text_normalization.zh.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. twelve thirty -> time { hours: "12" minutes: "30" }
        e.g. twelve past one -> time { minutes: "12" hours: "1" }
        e.g. two o clock a m -> time { hours: "2" suffix: "a.m." }
        e.g. quarter to two -> time { hours: "1" minutes: "45" }
        e.g. quarter past two -> time { hours: "2" minutes: "15" }
        e.g. half past two -> time { hours: "2" minutes: "30" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")
        # hours, minutes, seconds, suffix, zone, style, speak_period

        noon_graph = pynini.string_file(get_abs_path("data/time/noon.tsv"))
        hour_graph = pynini.string_file(get_abs_path("data/time/hour.tsv"))
        minute_graph = pynini.string_file(get_abs_path("data/time/minute.tsv"))
        second_graph = pynini.string_file(get_abs_path("data/time/second.tsv"))

        final_graph = (
            (pynutil.insert('noon: "') + noon_graph + pynutil.insert('" ')).ques
            + pynutil.insert('hour: "')
            + hour_graph
            + pynutil.insert('"')
            + pynutil.insert(' minute: "')
            + minute_graph
            + pynutil.delete("分").ques
            + pynutil.insert('"')
            + (pynutil.insert(' second: "') + second_graph + pynutil.insert('"')).ques
        )

        final_graph = self.add_tokens(final_graph.optimize())

        self.fst = final_graph.optimize()
