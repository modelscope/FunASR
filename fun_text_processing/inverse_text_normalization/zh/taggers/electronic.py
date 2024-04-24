import pynini
from fun_text_processing.inverse_text_normalization.zh.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.zh.graph_utils import (
    DAMO_ALPHA,
    GraphFst,
    insert_space,
)
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. c d f one at a b c dot e d u -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }
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
        process_dot = pynini.cross("点", ".")
        username = (
            alpha_num + pynini.closure(delete_extra_space + accepted_username)
        ) | pynutil.add_weight(pynini.closure(DAMO_ALPHA, 1), weight=0.0001)
        username = pynutil.insert('username: "') + username + pynutil.insert('"')
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
            + (pynutil.delete("at") | pynutil.delete("艾特"))
            + insert_space
            + delete_extra_space
            + domain_graph
        )

        ############# url ###
        protocol_end = pynini.cross(pynini.union("w w w", "www"), "www")
        protocol_start = (
            pynini.cross("h t t p", "http") | pynini.cross("h t t p s", "https")
        ) + pynini.cross("冒号斜杆斜杠", "://")
        # .com,
        ending = (
            delete_extra_space
            + symbols
            + delete_extra_space
            + (
                domain
                | pynini.closure(
                    accepted_username + delete_extra_space,
                )
                + accepted_username
            )
        )

        protocol_default = (
            (
                (pynini.closure(delete_extra_space + accepted_username, 1) | server)
                | pynutil.add_weight(pynini.closure(DAMO_ALPHA, 1), weight=0.0001)
            )
            + pynini.closure(ending, 1)
        ).optimize()
        protocol = (
            pynini.closure(protocol_start, 0, 1)
            + protocol_end
            + pynini.closure(pynini.cross(" ", ""), 0, 1)
            + process_dot
            + pynini.closure(pynini.cross(" ", ""), 0, 1)
            + protocol_default
        ).optimize()

        protocol |= (
            pynini.closure(protocol_end + delete_extra_space + process_dot, 0, 1) + protocol_default
        )

        protocol = pynutil.insert('protocol: "') + protocol.optimize() + pynutil.insert('"')
        graph |= protocol
        ########

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
