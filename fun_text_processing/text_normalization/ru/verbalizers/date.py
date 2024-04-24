import pynini
from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from fun_text_processing.text_normalization.ru.alphabet import RU_ALPHA
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        tokens { date { day: "первое мая" } } -> "первое мая"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete('day: "') + pynini.closure(RU_ALPHA | " ", 1) + pynutil.delete('"')
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
