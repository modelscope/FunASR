import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.ru.alphabet import RU_ALPHA, TO_CYRILLIC
from fun_text_processing.text_normalization.ru.utils import get_abs_path
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: email addresses
        e.g. "ab@nd.ru" -> electronic { username: "эй би собака эн ди точка ру" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        # tagger
        accepted_symbols = []
        with open(get_abs_path("data/electronic/symbols.tsv"), "r", encoding="utf-8") as f:
            for line in f:
                symbol, _ = line.split("\t")
                accepted_symbols.append(pynini.accep(symbol))
        username = (
            pynutil.insert('username: "')
            + DAMO_ALPHA
            + pynini.closure(DAMO_ALPHA | DAMO_DIGIT | pynini.union(*accepted_symbols))
            + pynutil.insert('"')
            + pynini.cross("@", " ")
        )
        domain_graph = (
            DAMO_ALPHA
            + (pynini.closure(DAMO_ALPHA | DAMO_DIGIT | pynini.accep("-") | pynini.accep(".")))
            + (DAMO_ALPHA | DAMO_DIGIT)
        )
        domain_graph = pynutil.insert('domain: "') + domain_graph + pynutil.insert('"')
        tagger_graph = (username + domain_graph).optimize()

        # verbalizer
        graph_digit = pynini.string_file(
            get_abs_path("data/numbers/digits_nominative_case.tsv")
        ).optimize()
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).optimize()
        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete('"')
            + (
                pynini.closure(
                    pynutil.add_weight(graph_digit + insert_space, 1.09)
                    | pynutil.add_weight(pynini.closure(graph_symbols + pynutil.insert(" ")), 1.09)
                    | pynutil.add_weight(DAMO_NOT_QUOTE + insert_space, 1.1)
                )
            )
            + pynutil.delete('"')
        )

        domain_default = (
            pynini.closure(DAMO_NOT_QUOTE + insert_space)
            + pynini.cross(".", "точка ")
            + DAMO_NOT_QUOTE
            + pynini.closure(insert_space + DAMO_NOT_QUOTE)
        )

        server_default = (
            pynini.closure((graph_digit | DAMO_ALPHA) + insert_space, 1)
            + pynini.closure(graph_symbols + insert_space)
            + pynini.closure((graph_digit | DAMO_ALPHA) + insert_space, 1)
        )
        server_common = (
            pynini.string_file(get_abs_path("data/electronic/server_name.tsv")) + insert_space
        )
        domain_common = pynini.cross(".", "точка ") + pynini.string_file(
            get_abs_path("data/electronic/domain.tsv")
        )
        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete('"')
            + (pynutil.add_weight(server_common, 1.09) | pynutil.add_weight(server_default, 1.1))
            + (pynutil.add_weight(domain_common, 1.09) | pynutil.add_weight(domain_default, 1.1))
            + delete_space
            + pynutil.delete('"')
        )

        graph = (
            user_name
            + delete_space
            + pynutil.insert("собака ")
            + delete_space
            + domain
            + delete_space
        )
        # replace all latin letters with their Ru verbalization
        verbalizer_graph = (
            graph.optimize() @ (pynini.closure(TO_CYRILLIC | RU_ALPHA | pynini.accep(" ")))
        ).optimize()
        verbalizer_graph = verbalizer_graph.optimize()

        self.final_graph = (tagger_graph @ verbalizer_graph).optimize()
        self.fst = self.add_tokens(
            pynutil.insert('username: "') + self.final_graph + pynutil.insert('"')
        ).optimize()
