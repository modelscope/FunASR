import pynini
from fun_text_processing.inverse_text_normalization.vi.graph_utils import (
    DAMO_ALPHA,
    GraphFst,
    insert_space,
)
from fun_text_processing.inverse_text_normalization.vi.utils import get_abs_path
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. c d f một a còng a b c dot e d u -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }
    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        delete_extra_space = pynutil.delete(" ")
        alpha_num = (
            DAMO_ALPHA
            | pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        )

        symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).invert()

        accepted_username = alpha_num | symbols
        process_dot = pynini.cross("chấm", ".")
        username = (
            pynutil.insert('username: "')
            + alpha_num
            + pynini.closure(delete_extra_space + accepted_username)
            + pynutil.insert('"')
        )
        single_alphanum = pynini.closure(alpha_num + delete_extra_space) + alpha_num
        server = single_alphanum | pynini.string_file(
            get_abs_path("data/electronic/server_name.tsv")
        )
        domain = single_alphanum | pynini.string_file(get_abs_path("data/electronic/domain.tsv"))
        multi_domain = (
            pynini.closure(process_dot + delete_extra_space + domain + delete_extra_space)
            + process_dot
            + delete_extra_space
            + domain
        )
        domain_graph = (
            pynutil.insert('domain: "')
            + server
            + delete_extra_space
            + multi_domain
            + pynutil.insert('"')
        )
        graph = (
            username
            + delete_extra_space
            + pynutil.delete(pynini.union("a còng", "a móc", "a vòng"))
            + insert_space
            + delete_extra_space
            + domain_graph
        )

        ############# url ###
        protocol_end = pynini.cross(pynini.union("w w w", "www"), "www")
        protocol_start = (
            pynini.cross("h t t p", "http") | pynini.cross("h t t p s", "https")
        ) + pynini.cross(" hai chấm sẹc sẹc ", "://")
        # .com,
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
            + pynini.closure(delete_extra_space + accepted_username, 1)
            + pynini.closure(ending, 1, 2)
        )
        protocol = pynutil.insert('protocol: "') + protocol + pynutil.insert('"')
        graph |= protocol
        ########

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
