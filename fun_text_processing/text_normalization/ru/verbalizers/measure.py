import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NON_BREAKING_SPACE,
    DAMO_SPACE,
    GraphFst,
    delete_space,
)
from fun_text_processing.text_normalization.ru.alphabet import RU_ALPHA
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, e.g.
        measure { cardinal { integer: "два килограма" } } -> "два килограма"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        graph = (
            pynutil.delete(' cardinal { integer: "')
            + pynini.closure(RU_ALPHA | DAMO_SPACE | DAMO_NON_BREAKING_SPACE)
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete("}")
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
