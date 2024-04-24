import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    DAMO_NOT_SPACE,
    DAMO_SIGMA,
    TO_UPPER,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.en.utils import get_abs_path
from pynini.examples import plurals
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. tokens { electronic { username: "cdf1" domain: "abc.edu" } } -> c d f one at a b c dot e d u

    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)
        graph_digit_no_zero = pynini.invert(
            pynini.string_file(get_abs_path("data/number/digit.tsv"))
        ).optimize()
        graph_zero = pynini.cross("0", "zero")

        if not deterministic:
            graph_zero |= pynini.cross("0", "o") | pynini.cross("0", "oh")

        graph_digit = graph_digit_no_zero | graph_zero
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbol.tsv")).optimize()

        default_chars_symbols = pynini.cdrewrite(
            pynutil.insert(" ") + (graph_symbols | graph_digit) + pynutil.insert(" "),
            "",
            "",
            DAMO_SIGMA,
        )
        default_chars_symbols = pynini.compose(
            pynini.closure(DAMO_NOT_SPACE), default_chars_symbols.optimize()
        ).optimize()

        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete('"')
            + default_chars_symbols
            + pynutil.delete('"')
        )

        domain_common = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

        domain = (
            default_chars_symbols
            + insert_space
            + plurals._priority_union(
                domain_common,
                pynutil.add_weight(pynini.cross(".", "dot"), weight=0.0001),
                DAMO_SIGMA,
            )
            + pynini.closure(
                insert_space
                + (pynini.cdrewrite(TO_UPPER, "", "", DAMO_SIGMA) @ default_chars_symbols),
                0,
                1,
            )
        )
        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete('"')
            + domain
            + delete_space
            + pynutil.delete('"')
        ).optimize()

        protocol = (
            pynutil.delete('protocol: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )
        graph = (
            pynini.closure(protocol + delete_space, 0, 1)
            + pynini.closure(user_name + delete_space + pynutil.insert(" at ") + delete_space, 0, 1)
            + domain
            + delete_space
        ).optimize() @ pynini.cdrewrite(delete_extra_space, "", "", DAMO_SIGMA)

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
