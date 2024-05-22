import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_CHAR,
    DAMO_DIGIT,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
)
from fun_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

ordinal_exceptions = pynini.string_file(get_abs_path("data/fractions/ordinal_exceptions.tsv"))
higher_powers_of_ten = pynini.string_file(get_abs_path("data/fractions/powers_of_ten.tsv"))


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "23 4/5" ->
    tokens { fraction { integer: "veintitrés" numerator: "cuatro" denominator: "quinto" mophosyntactic_features: "ordinal" } }

    Args:
        cardinal: CardinalFst
        ordinal: OrdinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph
        ordinal_graph = ordinal.graph

        # 2-10 are all ordinals
        three_to_ten = pynini.string_map(
            [
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
            ]
        )
        block_three_to_ten = pynutil.delete(three_to_ten)  # To block cardinal productions
        if not deterministic:  # Multiples of tens are sometimes rendered as ordinals
            three_to_ten |= pynini.string_map(
                [
                    "20",
                    "30",
                    "40",
                    "50",
                    "60",
                    "70",
                    "80",
                    "90",
                ]
            )
        graph_three_to_ten = three_to_ten @ ordinal_graph
        graph_three_to_ten @= pynini.cdrewrite(ordinal_exceptions, "", "", DAMO_SIGMA)

        # Higher powers of tens (and multiples) are converted to ordinals.
        hundreds = pynini.string_map(
            [
                "100",
                "200",
                "300",
                "400",
                "500",
                "600",
                "700",
                "800",
                "900",
            ]
        )
        graph_hundreds = hundreds @ ordinal_graph

        multiples_of_thousand = ordinal.multiples_of_thousand  # So we can have X milésimos

        graph_higher_powers_of_ten = (
            pynini.closure(ordinal.one_to_one_thousand + DAMO_SPACE, 0, 1)
            + pynini.closure("mil ", 0, 1)
            + pynini.closure(ordinal.one_to_one_thousand + DAMO_SPACE, 0, 1)
        )  # x millones / x mil millones / x mil z millones
        graph_higher_powers_of_ten += higher_powers_of_ten
        graph_higher_powers_of_ten = cardinal_graph @ graph_higher_powers_of_ten
        graph_higher_powers_of_ten @= pynini.cdrewrite(
            pynutil.delete("un "),
            pynini.accep("[BOS]"),
            pynini.project(higher_powers_of_ten, "output"),
            DAMO_SIGMA,
        )  # we drop 'un' from these ordinals (millionths, not one-millionths)

        graph_higher_powers_of_ten = (
            multiples_of_thousand | graph_hundreds | graph_higher_powers_of_ten
        )
        block_higher_powers_of_ten = pynutil.delete(
            pynini.project(graph_higher_powers_of_ten, "input")
        )  # For cardinal graph

        graph_fractions_ordinals = graph_higher_powers_of_ten | graph_three_to_ten
        graph_fractions_ordinals += pynutil.insert(
            '" morphosyntactic_features: "ordinal"'
        )  # We note the root for processing later

        # Blocking the digits and hundreds from Cardinal graph
        graph_fractions_cardinals = pynini.cdrewrite(
            block_three_to_ten | block_higher_powers_of_ten,
            pynini.accep("[BOS]"),
            pynini.accep("[EOS]"),
            DAMO_SIGMA,
        )
        graph_fractions_cardinals @= DAMO_CHAR.plus @ pynini.cdrewrite(
            pynutil.delete("0"), pynini.accep("[BOS]"), pynini.accep("[EOS]"), DAMO_SIGMA
        )  # Empty characters become '0' for DAMO_CHAR fst, so need to block
        graph_fractions_cardinals @= cardinal_graph
        graph_fractions_cardinals += pynutil.insert(
            '" morphosyntactic_features: "add_root"'
        )  # blocking these entries to reduce erroneous possibilities in debugging

        if deterministic:
            graph_fractions_cardinals = (
                pynini.closure(DAMO_DIGIT, 1, 2) @ graph_fractions_cardinals
            )  # Past hundreds the conventional scheme can be hard to read. For determinism we stop here

        graph_denominator = pynini.union(
            graph_fractions_ordinals,
            graph_fractions_cardinals,
            pynutil.add_weight(cardinal_graph + pynutil.insert('"'), 0.001),
        )  # Last form is simply recording the cardinal. Weighting so last resort

        integer = (
            pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"') + DAMO_SPACE
        )
        numerator = (
            pynutil.insert('numerator: "')
            + cardinal_graph
            + (pynini.cross("/", '" ') | pynini.cross(" / ", '" '))
        )
        denominator = pynutil.insert('denominator: "') + graph_denominator

        self.graph = pynini.closure(integer, 0, 1) + numerator + denominator

        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
