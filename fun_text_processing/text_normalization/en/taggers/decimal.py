import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_SIGMA,
    TO_UPPER,
    GraphFst,
    get_abs_path,
)
from pynini.lib import pynutil

delete_space = pynutil.delete(" ")
quantities = pynini.string_file(get_abs_path("data/number/thousand.tsv"))
quantities_abbr = pynini.string_file(get_abs_path("data/number/quantity_abbr.tsv"))
quantities_abbr |= TO_UPPER @ quantities_abbr


def get_quantity(
    decimal: "pynini.FstLike", cardinal_up_to_hundred: "pynini.FstLike", include_abbr: bool
) -> "pynini.FstLike":
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 1 million -> integer_part: "one" quantity: "million"
    e.g. 1.5 million -> integer_part: "one" fractional_part: "five" quantity: "million"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    quantity_wo_thousand = pynini.project(quantities, "input") - pynini.union("k", "K", "thousand")
    if include_abbr:
        quantity_wo_thousand |= pynini.project(quantities_abbr, "input") - pynini.union(
            "k", "K", "thousand"
        )
    res = (
        pynutil.insert('integer_part: "')
        + cardinal_up_to_hundred
        + pynutil.insert('"')
        + pynini.closure(pynutil.delete(" "), 0, 1)
        + pynutil.insert(' quantity: "')
        + (quantity_wo_thousand @ (quantities | quantities_abbr))
        + pynutil.insert('"')
    )
    if include_abbr:
        quantity = quantities | quantities_abbr
    else:
        quantity = quantities
    res |= (
        decimal
        + pynini.closure(pynutil.delete(" "), 0, 1)
        + pynutil.insert('quantity: "')
        + quantity
        + pynutil.insert('"')
    )
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        -12.5006 billion -> decimal { negative: "true" integer_part: "12"  fractional_part: "five o o six" quantity: "billion" }
        1 billion -> decimal { integer_part: "one" quantity: "billion" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph_with_and
        cardinal_graph_hundred_component_at_least_one_none_zero_digit = (
            cardinal.graph_hundred_component_at_least_one_none_zero_digit
        )

        self.graph = cardinal.single_digits_graph.optimize()

        if not deterministic:
            self.graph = self.graph | cardinal_graph

        point = pynutil.delete(".")
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1
        )

        self.graph_fractional = (
            pynutil.insert('fractional_part: "') + self.graph + pynutil.insert('"')
        )
        self.graph_integer = (
            pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        )
        final_graph_wo_sign = (
            pynini.closure(self.graph_integer + pynutil.insert(" "), 0, 1)
            + point
            + pynutil.insert(" ")
            + self.graph_fractional
        )

        quantity_w_abbr = get_quantity(
            final_graph_wo_sign,
            cardinal_graph_hundred_component_at_least_one_none_zero_digit,
            include_abbr=True,
        )
        quantity_wo_abbr = get_quantity(
            final_graph_wo_sign,
            cardinal_graph_hundred_component_at_least_one_none_zero_digit,
            include_abbr=False,
        )
        self.final_graph_wo_negative_w_abbr = final_graph_wo_sign | quantity_w_abbr
        self.final_graph_wo_negative = final_graph_wo_sign | quantity_wo_abbr

        # reduce options for non_deterministic and allow either "oh" or "zero", but not combination
        if not deterministic:
            no_oh_zero = pynini.difference(
                DAMO_SIGMA,
                (DAMO_SIGMA + "oh" + DAMO_SIGMA + "zero" + DAMO_SIGMA)
                | (DAMO_SIGMA + "zero" + DAMO_SIGMA + "oh" + DAMO_SIGMA),
            ).optimize()
            no_zero_oh = pynini.difference(
                DAMO_SIGMA,
                DAMO_SIGMA + pynini.accep("zero") + DAMO_SIGMA + pynini.accep("oh") + DAMO_SIGMA,
            ).optimize()

            self.final_graph_wo_negative |= pynini.compose(
                self.final_graph_wo_negative,
                pynini.cdrewrite(
                    pynini.cross('integer_part: "zero"', 'integer_part: "oh"'),
                    DAMO_SIGMA,
                    DAMO_SIGMA,
                    DAMO_SIGMA,
                ),
            )
            self.final_graph_wo_negative = pynini.compose(
                self.final_graph_wo_negative, no_oh_zero
            ).optimize()
            self.final_graph_wo_negative = pynini.compose(
                self.final_graph_wo_negative, no_zero_oh
            ).optimize()

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
