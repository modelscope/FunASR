import pynini
from fun_text_processing.text_normalization.de.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import GraphFst, insert_space
from pynini.lib import pynutil

quantities = pynini.string_file(get_abs_path("data/numbers/quantities.tsv"))


def get_quantity(
    decimal: "pynini.FstLike", cardinal_up_to_hundred: "pynini.FstLike"
) -> "pynini.FstLike":
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 1 million -> integer_part: "eine" quantity: "million"
    e.g. 1.4 million -> integer_part: "eins" fractional_part: "vier" quantity: "million"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = cardinal_up_to_hundred

    res = (
        pynutil.insert('integer_part: "')
        + numbers
        + pynutil.insert('"')
        + pynini.accep(" ")
        + pynutil.insert('quantity: "')
        + quantities
        + pynutil.insert('"')
    )
    res |= (
        decimal
        + pynini.accep(" ")
        + pynutil.insert('quantity: "')
        + quantities
        + pynutil.insert('"')
    )
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        -11,4006 billion -> decimal { negative: "true" integer_part: "elf"  fractional_part: "vier null null sechs" quantity: "billion" preserve_order: true }
        1 billion -> decimal { integer_part: "eins" quantity: "billion" preserve_order: true }
    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_digit |= pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit |= pynini.cross("1", "eins")
        self.graph = graph_digit + pynini.closure(insert_space + graph_digit).optimize()

        point = pynutil.delete(",")
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1
        )

        self.graph_fractional = (
            pynutil.insert('fractional_part: "') + self.graph + pynutil.insert('"')
        )
        self.graph_integer = (
            pynutil.insert('integer_part: "') + cardinal.graph + pynutil.insert('"')
        )
        final_graph_wo_sign = self.graph_integer + point + insert_space + self.graph_fractional

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign, cardinal.graph_hundred_component_at_least_one_none_zero_digit
        )
        final_graph = optional_graph_negative + self.final_graph_wo_negative
        final_graph += pynutil.insert(" preserve_order: true")

        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
