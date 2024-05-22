import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import (
    DAMO_ALPHA,
    GraphFst,
    insert_space,
)
from fun_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying 'electronic' semiotic classes, i.e.
    email address (which get converted to "username" and "domain" fields),
    and URLS (which get converted to a "protocol" field).
        e.g. c d f une arobase a b c point e d u -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }
        e.g. double vé double vé double vé a b c point e d u -> tokens { electronic { protocol: "www.abc.edu" } }
    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        delete_extra_space = pynutil.delete(" ")
        alpha_num = (
            DAMO_ALPHA
            | pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        )

        symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv"))
        ampersand = pynini.string_map([("arobase"), ("chez"), ("at"), ("à")])

        accepted_username = alpha_num | symbols
        process_dot = pynini.cross("point", ".")
        username = (
            pynutil.insert('username: "')
            + alpha_num
            + delete_extra_space
            + pynini.closure(accepted_username + delete_extra_space)
            + alpha_num
            + pynutil.insert('"')
        )
        single_alphanum = pynini.closure(alpha_num + delete_extra_space) + alpha_num
        server = single_alphanum | pynini.string_file(
            get_abs_path("data/electronic/server_name.tsv")
        )
        domain = single_alphanum | pynini.string_file(get_abs_path("data/electronic/domain.tsv"))
        domain_graph = (
            pynutil.insert('domain: "')
            + server
            + delete_extra_space
            + process_dot
            + delete_extra_space
            + domain
            + pynutil.insert('"')
        )
        graph = (
            username
            + delete_extra_space
            + pynutil.delete(ampersand)
            + insert_space
            + delete_extra_space
            + domain_graph
        )

        ############# url ###
        protocol_end = pynini.cross(
            pynini.union("www", "w w w", "double vé double vé double vé"), "www"
        )
        protocol_start = pynini.cross(pynini.union("http", "h t t p", "ache té té pé"), "http")
        protocol_start |= pynini.cross(
            pynini.union("https", "h t t p s", "ache té té pé esse"), "https"
        )
        protocol_start += pynini.cross(
            pynini.union(
                " deux-points barre oblique barre oblique ",
                " deux-points barre barre ",
                " deux-points double barre ",
                " deux-points slash slash ",
            ),
            "://",
        )

        # e.g. .com, .es
        ending = (
            delete_extra_space
            + symbols
            + delete_extra_space
            + (domain | pynini.closure(accepted_username + delete_extra_space) + accepted_username)
        )

        protocol = (
            pynini.closure(protocol_start, 0, 1)
            + protocol_end
            + delete_extra_space
            + process_dot
            + delete_extra_space
            + (pynini.closure(delete_extra_space + accepted_username, 1) | server)
            + pynini.closure(ending, 1)
        )
        protocol = pynutil.insert('protocol: "') + protocol + pynutil.insert('"')
        graph |= protocol
        ########

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
