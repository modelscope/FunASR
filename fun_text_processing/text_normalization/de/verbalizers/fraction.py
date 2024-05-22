import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    GraphFst,
    delete_preserve_order,
    insert_space,
)
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
        e.g. fraction { integer: "drei" numerator: "eins" denominator: "zwei" }-> drei ein halb
        e.g. fraction { numerator: "vier" denominator: "zwei" } -> vier halbe
        e.g. fraction { numerator: "drei" denominator: "vier" } -> drei viertel

    Args:
        ordinal: ordinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.closure(
            pynini.cross('negative: "true"', "minus ") + pynutil.delete(" "), 0, 1
        )
        change_one = pynini.cdrewrite(
            pynutil.add_weight(pynini.cross("eins", "ein"), weight=-0.0001),
            "[BOS]",
            "[EOS]",
            DAMO_SIGMA,
        )
        change_numerator_two = pynini.cdrewrite(
            pynini.cross("zweitel", "halbe"), "[BOS]", "[EOS]", DAMO_SIGMA
        )
        integer = pynutil.delete('integer_part: "') + change_one + pynutil.delete('" ')
        numerator = pynutil.delete('numerator: "') + change_one + pynutil.delete('" ')
        denominator = (
            pynutil.delete('denominator: "')
            + pynini.closure(DAMO_NOT_QUOTE)
            @ (
                pynini.cdrewrite(
                    pynini.closure(ordinal.ordinal_stem, 0, 1), "", "[EOS]", DAMO_SIGMA
                )
                + pynutil.insert("tel")
            )
            @ change_numerator_two
            + pynutil.delete('"')
        )

        integer += insert_space + pynini.closure(pynutil.insert("und ", weight=0.001), 0, 1)

        denominator_one_half = pynini.cdrewrite(
            pynini.cross("ein halbe", "ein halb"), "[BOS]", "[EOS]", DAMO_SIGMA
        )

        fraction_default = (numerator + insert_space + denominator) @ denominator_one_half

        self.graph = optional_sign + pynini.closure(integer, 0, 1) + fraction_default

        graph = self.graph + delete_preserve_order

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
