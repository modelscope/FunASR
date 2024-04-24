import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    convert_space,
    insert_space,
)
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
        null vier eins eins eins zwei drei vier eins zwei drei vier -> tokens { name: "(0411) 1234-1234" }

    Args:
        tn_cardinal_tagger: TN Cardinal Tagger
    """

    def __init__(self, tn_cardinal_tagger: GraphFst, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)
        separator = pynini.accep(" ")  # between components
        digit = pynini.union(*list(map(str, range(1, 10)))) @ tn_cardinal_tagger.two_digit_non_zero
        zero = pynini.cross("0", "null")

        number_part = (
            pynutil.delete("(")
            + zero
            + insert_space
            + pynini.closure(digit + insert_space, 2, 2)
            + digit
            + pynutil.delete(")")
            + separator
            + pynini.closure(digit + insert_space, 3, 3)
            + digit
            + pynutil.delete("-")
            + insert_space
            + pynini.closure(digit + insert_space, 3, 3)
            + digit
        )
        graph = convert_space(pynini.invert(number_part))
        final_graph = pynutil.insert('name: "') + graph + pynutil.insert('"')

        self.fst = final_graph.optimize()
