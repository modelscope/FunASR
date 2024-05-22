import pynini
from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from fun_text_processing.text_normalization.ru.alphabet import RU_ALPHA
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. electronic { username: "эй би собака эн ди точка ру" } -> "эй би собака эн ди точка ру"

    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete('username: "') + pynini.closure(RU_ALPHA | " ") + pynutil.delete('"')
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
