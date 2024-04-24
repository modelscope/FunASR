import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_CHAR, GraphFst, convert_space
from fun_text_processing.text_normalization.ru.alphabet import RU_ALPHA, TO_CYRILLIC
from fun_text_processing.text_normalization.ru.utils import get_abs_path, load_labels
from pynini.lib import pynutil


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        misses -> tokens { name: "mrs" }
        for non-deterministic case: "Dr. Abc" ->
            tokens { name: "drive" } tokens { name: "Abc" }
            tokens { name: "doctor" } tokens { name: "Abc" }
            tokens { name: "Dr." } tokens { name: "Abc" }
    This class has highest priority among all classifier grammars. Whitelisted tokens are defined and loaded from "data/whitelist.tsv".

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        input_file: path to a file with whitelist replacements
    """

    def __init__(self, input_case: str, deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        def _get_whitelist_graph(input_case, file):
            whitelist = load_labels(file)
            if input_case == "lower_cased":
                whitelist = [[x[0].lower()] + x[1:] for x in whitelist]
            else:
                whitelist = [[x[0].lower()] + x[1:] for x in whitelist]
            graph = pynini.string_map(whitelist)
            return graph

        graph = _get_whitelist_graph(input_case, get_abs_path("data/whitelist.tsv"))

        if input_file:
            graph = _get_whitelist_graph(input_case, input_file)

        units_graph = _get_whitelist_graph(input_case, file=get_abs_path("data/measurements.tsv"))
        # do not replace single letter units, like `м`, `°` and `%` will be replaced
        units_graph = pynini.compose(
            (DAMO_CHAR ** (2, ...) | pynini.difference(DAMO_CHAR, RU_ALPHA)), units_graph
        )
        graph |= units_graph.optimize()
        graph |= TO_CYRILLIC + pynini.closure(pynutil.insert(" ") + TO_CYRILLIC)

        self.final_graph = convert_space(graph)
        self.fst = (pynutil.insert('name: "') + self.final_graph + pynutil.insert('"')).optimize()
