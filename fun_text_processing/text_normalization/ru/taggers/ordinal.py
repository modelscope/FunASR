# Adapted from https://github.com/google/TextNormalizationCoveringGrammars
# Russian minimally supervised number grammar.


import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_SIGMA, GraphFst
from fun_text_processing.text_normalization.ru.utils import get_abs_path
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "2" -> ordinal { integer: "второе" } }

    Args:
        number_names: number_names for cardinal and ordinal numbers
        alternative_formats: alternative format for cardinal and ordinal numbers
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, number_names: dict, alternative_formats: dict, deterministic=False):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        one_thousand_alternative = alternative_formats["one_thousand_alternative"]
        separators = alternative_formats["separators"]

        ordinal = number_names["ordinal_number_names"]

        ordinal |= ordinal @ one_thousand_alternative
        ordinal_numbers = separators @ ordinal

        # to handle cases like 2-ая
        endings = pynini.string_file(get_abs_path("data/numbers/ordinal_endings.tsv"))
        not_dash = pynini.closure(pynini.difference(DAMO_SIGMA, "-"))
        del_ending = pynini.cdrewrite(pynini.cross("-" + not_dash, ""), "", "[EOS]", DAMO_SIGMA)
        ordinal_numbers_marked = (
            ((separators @ ordinal).optimize() + pynini.accep("-") + not_dash).optimize()
            @ (DAMO_SIGMA + endings).optimize()
            @ del_ending
        ).optimize()

        self.ordinal_numbers = ordinal_numbers
        # "03" -> remove leading zeros and verbalize
        leading_zeros = pynini.closure(pynini.cross("0", ""))
        self.ordinal_numbers_with_leading_zeros = (leading_zeros + ordinal_numbers).optimize()

        final_graph = (ordinal_numbers | ordinal_numbers_marked).optimize()
        final_graph = pynutil.insert('integer: "') + final_graph + pynutil.insert('"')
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
