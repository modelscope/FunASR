import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_SIGMA, GraphFst
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. acht uhr e s t-> time { hours: "8" zone: "e s t" }
        e.g. dreizehn uhr -> time { hours: "13" }
        e.g. dreizehn uhr zehn -> time { hours: "13" minutes: "10" }
        e.g. viertel vor zwölf -> time { minutes: "45" hours: "11" }
        e.g. viertel nach zwölf -> time { minutes: "15" hours: "12" }
        e.g. halb zwölf -> time { minutes: "30" hours: "11" }
        e.g. drei vor zwölf -> time { minutes: "57" hours: "11" }
        e.g. drei nach zwölf -> time { minutes: "3" hours: "12" }
        e.g. drei uhr zehn minuten zehn sekunden -> time { hours: "3" hours: "10" sekunden: "10"}

    Args:
        tn_time_verbalizer: TN time verbalizer
    """

    def __init__(self, tn_time_verbalizer: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)
        # lazy way to make sure compounds work
        optional_delete_space = pynini.closure(DAMO_SIGMA | pynutil.delete(" ", weight=0.0001))
        graph = (tn_time_verbalizer.graph @ optional_delete_space).invert().optimize()
        self.fst = self.add_tokens(graph).optimize()
