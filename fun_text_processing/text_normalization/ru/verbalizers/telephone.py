import pynini
from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from fun_text_processing.text_normalization.ru.alphabet import RU_ALPHA
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone, e.g.
        telephone { number_part: "восемь девятьсот тринадцать девятьсот восемьдесят три пятьдесят шесть ноль один" } -> "восемь девятьсот тринадцать девятьсот восемьдесят три пятьдесят шесть ноль один"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="verbalize", deterministic=deterministic)

        graph = (
            pynutil.delete('number_part: "')
            + pynini.closure(RU_ALPHA | " ", 1)
            + pynutil.delete('"')
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
