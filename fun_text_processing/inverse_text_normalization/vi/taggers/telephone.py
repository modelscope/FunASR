import pynini
from fun_text_processing.inverse_text_normalization.vi.graph_utils import GraphFst, delete_space
from fun_text_processing.inverse_text_normalization.vi.utils import get_abs_path
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
        một hai ba một hai ba năm sáu bảy tám -> { number_part: "1231235678" }
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        digit = graph_digit | graph_zero
        last_digit = (
            digit | pynini.cross("mốt", "1") | pynini.cross("tư", "4") | pynini.cross("lăm", "5")
        )

        graph_number_part = pynini.closure(digit + delete_space, 2) + last_digit
        number_part = pynutil.insert('number_part: "') + graph_number_part + pynutil.insert('"')

        graph = number_part
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
