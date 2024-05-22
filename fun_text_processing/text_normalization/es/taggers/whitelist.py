import pynini
from fun_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space
from fun_text_processing.text_normalization.es.utils import get_abs_path, load_labels
from pynini.lib import pynutil


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        "sr." -> tokens { name: "señor" }
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
            graph = pynini.string_map(whitelist)
            return graph

        graph = _get_whitelist_graph(input_case, get_abs_path("data/whitelist.tsv"))
        if not deterministic and input_case != "lower_cased":
            graph |= pynutil.add_weight(
                _get_whitelist_graph("lower_cased", get_abs_path("data/whitelist.tsv")),
                weight=0.0001,
            )

        if input_file:
            whitelist_provided = _get_whitelist_graph(input_case, input_file)
            if not deterministic:
                graph |= whitelist_provided
            else:
                graph = whitelist_provided

        if not deterministic:
            units_graph = _get_whitelist_graph(
                input_case, file=get_abs_path("data/measures/measurements.tsv")
            )
            graph |= units_graph

        self.graph = graph
        self.final_graph = convert_space(self.graph).optimize()
        self.fst = (pynutil.insert('name: "') + self.final_graph + pynutil.insert('"')).optimize()
