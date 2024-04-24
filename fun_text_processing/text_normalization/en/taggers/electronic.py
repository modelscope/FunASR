import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_SIGMA,
    GraphFst,
    get_abs_path,
    insert_space,
)
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. cdf1@abc.edu -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        accepted_symbols = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/symbol.tsv")), "input"
        )
        accepted_common_domains = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/domain.tsv")), "input"
        )
        all_accepted_symbols = DAMO_ALPHA + pynini.closure(
            DAMO_ALPHA | DAMO_DIGIT | accepted_symbols
        )
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbol.tsv")).optimize()

        username = (
            pynutil.insert('username: "')
            + all_accepted_symbols
            + pynutil.insert('"')
            + pynini.cross("@", " ")
        )
        domain_graph = all_accepted_symbols + pynini.accep(".") + all_accepted_symbols + DAMO_ALPHA
        protocol_symbols = pynini.closure(
            (graph_symbols | pynini.cross(":", "semicolon")) + pynutil.insert(" ")
        )
        protocol_start = (pynini.cross("https", "HTTPS ") | pynini.cross("http", "HTTP ")) + (
            pynini.accep("://") @ protocol_symbols
        )
        protocol_file_start = (
            pynini.accep("file") + insert_space + (pynini.accep(":///") @ protocol_symbols)
        )

        protocol_end = pynini.cross("www", "WWW ") + pynini.accep(".") @ protocol_symbols
        protocol = (
            protocol_file_start | protocol_start | protocol_end | (protocol_start + protocol_end)
        )

        domain_graph = (
            pynutil.insert('domain: "')
            + pynini.difference(domain_graph, pynini.project(protocol, "input") + DAMO_SIGMA)
            + pynutil.insert('"')
        )
        domain_common_graph = (
            pynutil.insert('domain: "')
            + pynini.difference(
                all_accepted_symbols
                + accepted_common_domains
                + pynini.closure(
                    accepted_symbols + pynini.closure(DAMO_ALPHA | DAMO_DIGIT | accepted_symbols),
                    0,
                    1,
                ),
                pynini.project(protocol, "input") + DAMO_SIGMA,
            )
            + pynutil.insert('"')
        )

        protocol = pynutil.insert('protocol: "') + protocol + pynutil.insert('"')
        # email
        graph = username + domain_graph
        # abc.com, abc.com/123-sm
        graph |= domain_common_graph
        # www.abc.com/sdafsdf, or https://www.abc.com/asdfad or www.abc.abc/asdfad
        graph |= protocol + pynutil.insert(" ") + domain_graph

        final_graph = self.add_tokens(graph)

        self.fst = final_graph.optimize()
