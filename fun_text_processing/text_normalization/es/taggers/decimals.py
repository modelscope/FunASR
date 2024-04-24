import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.es.graph_utils import (
    cardinal_separator,
    decimal_separator,
    strip_cardinal_apocope,
)
from fun_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

quantities = pynini.string_file(get_abs_path("data/numbers/quantities.tsv"))
digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))


def get_quantity(
    decimal_graph: "pynini.FstLike", cardinal_graph: "pynini.FstLike"
) -> "pynini.FstLike":
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 2 millones -> integer_part: "dos" quantity: "millones"
    e.g. 2,4 millones -> integer_part: "dos" fractional_part: "quatro" quantity: "millones"
    e.g. 2,400 millones -> integer_part: "dos mil cuatrocientos" fractional_part: "quatro" quantity: "millones"

    Args:
        decimal_graph: DecimalFST
        cardinal_graph: CardinalFST
    """
    numbers = pynini.closure(DAMO_DIGIT, 1, 6) @ cardinal_graph
    numbers = pynini.cdrewrite(pynutil.delete(cardinal_separator), "", "", DAMO_SIGMA) @ numbers

    res = (
        pynutil.insert('integer_part: "')
        + numbers  # The cardinal we're passing only produces 'un' for one, so gender agreement is safe (all quantities are masculine). Limit to 10^6 power.
        + pynutil.insert('"')
        + DAMO_SPACE
        + pynutil.insert('quantity: "')
        + quantities
        + pynutil.insert('"')
    )
    res |= (
        decimal_graph
        + DAMO_SPACE
        + pynutil.insert('quantity: "')
        + quantities
        + pynutil.insert('"')
    )
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        -11,4006 billones -> decimal { negative: "true" integer_part: "once"  fractional_part: "cuatro cero cero seis" quantity: "billones" preserve_order: true }
        1 billón -> decimal { integer_part: "un" quantity: "billón" preserve_order: true }
    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)
        graph_digit = digit | zero

        if not deterministic:
            graph = pynini.union(graph_digit, cardinal.hundreds, cardinal.tens)
            graph += pynini.closure(insert_space + graph)

        else:
            # General pattern seems to be 1-3 digits: map as cardinal, default to digits otherwise \
            graph = pynini.union(
                graph_digit,
                cardinal.tens,
                cardinal.hundreds,
                graph_digit + pynini.closure(insert_space + graph_digit, 3),
                zero
                + pynini.closure(insert_space + zero)
                + pynini.closure(insert_space + graph_digit),  # For cases such as "1,010"
            )

        # Need to strip apocope everywhere BUT end of string
        reverse_apocope = pynini.string_map([("un", "uno"), ("ún", "uno")])
        apply_reverse_apocope = pynini.cdrewrite(reverse_apocope, "", DAMO_SPACE, DAMO_SIGMA)
        graph @= apply_reverse_apocope

        # Technically decimals should be space delineated groups of three, e.g. (1,333 333). This removes any possible spaces
        strip_formatting = pynini.cdrewrite(delete_space, "", "", DAMO_SIGMA)
        graph = strip_formatting @ graph

        self.graph = graph.optimize()

        graph_separator = pynutil.delete(decimal_separator)
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1
        )

        self.graph_fractional = (
            pynutil.insert('fractional_part: "') + self.graph + pynutil.insert('"')
        )

        # Integer graph maintains apocope except for ones place
        graph_integer = (
            strip_cardinal_apocope(cardinal.graph)
            if deterministic
            else pynini.union(cardinal.graph, strip_cardinal_apocope(cardinal.graph))
        )  # Gives us forms w/ and w/o apocope
        self.graph_integer = pynutil.insert('integer_part: "') + graph_integer + pynutil.insert('"')
        final_graph_wo_sign = (
            self.graph_integer + graph_separator + insert_space + self.graph_fractional
        )

        self.final_graph_wo_negative = (
            final_graph_wo_sign | get_quantity(final_graph_wo_sign, cardinal.graph).optimize()
        )
        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
