import pynini
from pynini import cross
from pynini.lib.pynutil import delete, insert, add_weight
from fun_text_processing.inverse_text_normalization.ja.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.ja.graph_utils import DAMO_CHAR, GraphFst
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. thirteenth -> ordinal { integer: "13" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        ties = pynini.string_file(get_abs_path("data/ordinals/ties.tsv"))
        teen = pynini.string_file(get_abs_path("data/ordinals/teen.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        hundred_digit = pynini.string_file(get_abs_path("data/numbers/hundred_digit.tsv"))
        addzero = insert("0")
        tens = ties + addzero | (digit + delete("十") + (digit | addzero))
        hundred = (
            digit
            + delete("百")
            + (
                tens
                | teen
                | add_weight(zero + digit, 0.1)
                | add_weight(digit + addzero, 0.5)
                | add_weight(addzero**2, 1.0)
            )
        )
        hundred |= cross("百", "1") + (
            tens
            | teen
            | add_weight(zero + digit, 0.1)
            | add_weight(digit + addzero, 0.5)
            | add_weight(addzero**2, 1.0)
        )
        hundred |= hundred_digit

        ordinal = digit | teen | tens | hundred
        graph = pynini.closure(DAMO_CHAR, 1) + ordinal

        self.graph = graph @ cardinal_graph
        final_graph = pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
