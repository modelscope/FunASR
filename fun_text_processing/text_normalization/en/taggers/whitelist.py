import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_CHAR,
    DAMO_NOT_SPACE,
    DAMO_SIGMA,
    DAMO_UPPER,
    SINGULAR_TO_PLURAL,
    GraphFst,
    convert_space,
)
from fun_text_processing.text_normalization.en.taggers.roman import get_names
from fun_text_processing.text_normalization.en.utils import (
    augment_labels_with_punct_at_end,
    get_abs_path,
    load_labels,
)
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

        def _get_whitelist_graph(input_case, file, keep_punct_add_end: bool = False):
            whitelist = load_labels(file)
            if input_case == "lower_cased":
                whitelist = [[x.lower(), y] for x, y in whitelist]
            else:
                whitelist = [[x, y] for x, y in whitelist]

            if keep_punct_add_end:
                whitelist.extend(augment_labels_with_punct_at_end(whitelist))

            graph = pynini.string_map(whitelist)
            return graph

        graph = _get_whitelist_graph(input_case, get_abs_path("data/whitelist/tts.tsv"))
        graph |= pynini.compose(
            pynini.difference(DAMO_SIGMA, pynini.accep("/")).optimize(),
            _get_whitelist_graph(input_case, get_abs_path("data/whitelist/symbol.tsv")),
        ).optimize()

        if deterministic:
            names = get_names()
            graph |= (
                pynini.cross(pynini.union("st", "St", "ST"), "Saint")
                + pynini.closure(pynutil.delete("."))
                + pynini.accep(" ")
                + names
            )
        else:
            graph |= _get_whitelist_graph(
                input_case, get_abs_path("data/whitelist/alternatives.tsv"), keep_punct_add_end=True
            )

        for x in [".", ". "]:
            graph |= (
                DAMO_UPPER
                + pynini.closure(pynutil.delete(x) + DAMO_UPPER, 2)
                + pynini.closure(pynutil.delete("."), 0, 1)
            )

        if not deterministic:
            multiple_forms_whitelist_graph = get_formats(
                get_abs_path("data/whitelist/alternatives_all_format.tsv")
            )
            graph |= multiple_forms_whitelist_graph

            graph_unit = pynini.string_file(
                get_abs_path("data/measure/unit.tsv")
            ) | pynini.string_file(get_abs_path("data/measure/unit_alternatives.tsv"))
            graph_unit_plural = graph_unit @ SINGULAR_TO_PLURAL
            units_graph = pynini.compose(
                DAMO_CHAR ** (3, ...), convert_space(graph_unit | graph_unit_plural)
            )
            graph |= units_graph

        # convert to states only if comma is present before the abbreviation to avoid converting all caps words,
        # e.g. "IN", "OH", "OK"
        # TODO or only exclude above?
        states = load_labels(get_abs_path("data/address/state.tsv"))
        additional_options = []
        for x, y in states:
            if input_case == "lower_cased":
                x = x.lower()
            additional_options.append((x, f"{y[0]}.{y[1:]}"))
            if not deterministic:
                additional_options.append((x, f"{y[0]}.{y[1:]}."))

        states.extend(additional_options)
        state_graph = pynini.string_map(states)
        graph |= (
            pynini.closure(DAMO_NOT_SPACE, 1)
            + pynini.union(", ", ",")
            + pynini.invert(state_graph).optimize()
        )

        if input_file:
            whitelist_provided = _get_whitelist_graph(input_case, input_file)
            if not deterministic:
                graph |= whitelist_provided
            else:
                graph = whitelist_provided

        self.graph = (convert_space(graph)).optimize()

        self.fst = (pynutil.insert('name: "') + self.graph + pynutil.insert('"')).optimize()


def get_formats(input_f, input_case="cased", is_default=True):
    """
    Adds various abbreviation format options to the list of acceptable input forms
    """
    multiple_formats = load_labels(input_f)
    additional_options = []
    for x, y in multiple_formats:
        if input_case == "lower_cased":
            x = x.lower()
        additional_options.append(
            (f"{x}.", y)
        )  # default "dr" -> doctor, this includes period "dr." -> doctor
        additional_options.append(
            (f"{x[0].upper() + x[1:]}", f"{y[0].upper() + y[1:]}")
        )  # "Dr" -> Doctor
        additional_options.append(
            (f"{x[0].upper() + x[1:]}.", f"{y[0].upper() + y[1:]}")
        )  # "Dr." -> Doctor
    multiple_formats.extend(additional_options)

    if not is_default:
        multiple_formats = [
            (x, f"|raw_start|{x}|raw_end||norm_start|{y}|norm_end|") for (x, y) in multiple_formats
        ]

    multiple_formats = pynini.string_map(multiple_formats)
    return multiple_formats
