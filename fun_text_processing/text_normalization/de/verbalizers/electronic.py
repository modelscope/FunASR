import pynini
from fun_text_processing.text_normalization.de.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    GraphFst,
    delete_preserve_order,
    insert_space,
)
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. electronic { username: "abc" domain: "hotmail.com" } -> "a b c at hotmail punkt com"
                                                           -> "a b c at h o t m a i l punkt c o m"
                                                           -> "a b c at hotmail punkt c o m"
                                                           -> "a b c at h o t m a i l punkt com"
    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)
        graph_digit_no_zero = pynini.invert(
            pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        ).optimize() | pynini.cross("1", "eins")
        graph_zero = pynini.invert(
            pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        ).optimize()
        graph_digit = graph_digit_no_zero | graph_zero
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).optimize()
        server_common = pynini.string_file(get_abs_path("data/electronic/server_name.tsv"))
        domain_common = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

        def add_space_after_char():
            return pynini.closure(DAMO_NOT_QUOTE - pynini.accep(" ") + insert_space) + (
                DAMO_NOT_QUOTE - pynini.accep(" ")
            )

        verbalize_characters = pynini.cdrewrite(graph_symbols | graph_digit, "", "", DAMO_SIGMA)

        user_name = pynutil.delete('username: "') + add_space_after_char() + pynutil.delete('"')
        user_name @= verbalize_characters

        convert_defaults = (
            pynutil.add_weight(DAMO_NOT_QUOTE, weight=0.0001) | domain_common | server_common
        )
        domain = convert_defaults + pynini.closure(insert_space + convert_defaults)
        domain @= verbalize_characters

        domain = pynutil.delete('domain: "') + domain + pynutil.delete('"')
        protocol = (
            pynutil.delete('protocol: "')
            + add_space_after_char() @ pynini.cdrewrite(graph_symbols, "", "", DAMO_SIGMA)
            + pynutil.delete('"')
        )
        self.graph = (pynini.closure(protocol + pynini.accep(" "), 0, 1) + domain) | (
            user_name + pynini.accep(" ") + pynutil.insert("at ") + domain
        )
        delete_tokens = self.delete_tokens(self.graph + delete_preserve_order)
        self.fst = delete_tokens.optimize()
