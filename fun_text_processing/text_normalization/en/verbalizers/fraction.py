import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    GraphFst,
    insert_space,
)
from fun_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst
from pynini.examples import plurals
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
        e.g. tokens { fraction { integer: "twenty three" numerator: "four" denominator: "five" } } ->
        twenty three and four fifth

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)
        suffix = OrdinalFst().suffix

        integer = (
            pynutil.delete('integer_part: "')
            + pynini.closure(DAMO_NOT_QUOTE)
            + pynutil.delete('" ')
        )
        denominator_one = pynini.cross('denominator: "one"', "over one")
        denominator_half = pynini.cross('denominator: "two"', "half")
        denominator_quarter = pynini.cross('denominator: "four"', "quarter")

        denominator_rest = (
            pynutil.delete('denominator: "')
            + pynini.closure(DAMO_NOT_QUOTE) @ suffix
            + pynutil.delete('"')
        )

        denominators = plurals._priority_union(
            denominator_one,
            plurals._priority_union(
                denominator_half,
                plurals._priority_union(denominator_quarter, denominator_rest, DAMO_SIGMA),
                DAMO_SIGMA,
            ),
            DAMO_SIGMA,
        ).optimize()
        if not deterministic:
            denominators |= (
                pynutil.delete('denominator: "')
                + (pynini.accep("four") @ suffix)
                + pynutil.delete('"')
            )

        numerator_one = pynutil.delete('numerator: "') + pynini.accep("one") + pynutil.delete('" ')
        numerator_one = numerator_one + insert_space + denominators
        numerator_rest = (
            pynutil.delete('numerator: "')
            + (pynini.closure(DAMO_NOT_QUOTE) - pynini.accep("one"))
            + pynutil.delete('" ')
        )
        numerator_rest = numerator_rest + insert_space + denominators
        numerator_rest @= pynini.cdrewrite(
            plurals._priority_union(
                pynini.cross("half", "halves"), pynutil.insert("s"), DAMO_SIGMA
            ),
            "",
            "[EOS]",
            DAMO_SIGMA,
        )

        graph = numerator_one | numerator_rest

        conjunction = pynutil.insert("and ")
        if not deterministic and not lm:
            conjunction = pynini.closure(conjunction, 0, 1)

        integer = pynini.closure(integer + insert_space + conjunction, 0, 1)

        graph = integer + graph
        graph @= pynini.cdrewrite(
            pynini.cross("and one half", "and a half") | pynini.cross("over ones", "over one"),
            "",
            "[EOS]",
            DAMO_SIGMA,
        )

        self.graph = graph
        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
