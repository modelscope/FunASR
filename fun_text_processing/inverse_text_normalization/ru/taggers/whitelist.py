import pynini
from fun_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space
from fun_text_processing.text_normalization.ru.utils import get_abs_path
from pynini.lib import pynutil


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        "квартира" -> telephone { number_part: "кв." }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)
        whitelist = pynini.string_file(get_abs_path("data/whitelist.tsv")).invert()
        graph = pynutil.insert('name: "') + convert_space(whitelist) + pynutil.insert('"')
        self.fst = graph.optimize()
