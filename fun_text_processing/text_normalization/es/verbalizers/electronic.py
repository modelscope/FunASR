import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    GraphFst,
    delete_preserve_order,
    insert_space,
)
from fun_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

digit_no_zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))

graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv"))
server_common = pynini.string_file(get_abs_path("data/electronic/server_name.tsv"))
domain_common = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. electronic { username: "abc" domain: "hotmail.com" } -> "a b c arroba hotmail punto com"
                                                           -> "a b c arroba h o t m a i l punto c o m"
                                                           -> "a b c arroba hotmail punto c o m"
                                                           -> "a b c at h o t m a i l punto com"
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        graph_digit_no_zero = (
            digit_no_zero
            @ pynini.cdrewrite(pynini.cross("un", "uno"), "", "", DAMO_SIGMA).optimize()
        )
        graph_digit = graph_digit_no_zero | zero

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
            user_name + pynini.accep(" ") + pynutil.insert("arroba ") + domain
        )
        delete_tokens = self.delete_tokens(self.graph + delete_preserve_order)
        self.fst = delete_tokens.optimize()
