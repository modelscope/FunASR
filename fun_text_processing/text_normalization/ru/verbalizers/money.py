import pynini
from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from fun_text_processing.text_normalization.ru.alphabet import RU_ALPHA
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money {  "пять рублей" } -> пять рублей

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        graph = pynini.closure(RU_ALPHA | " ")
        delete_tokens = self.delete_tokens(
            pynutil.delete('integer_part: "') + graph + pynutil.delete('"')
        )
        self.fst = delete_tokens.optimize()
