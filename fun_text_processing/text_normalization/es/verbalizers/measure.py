import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    DAMO_SPACE,
    DAMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_preserve_order,
)
from fun_text_processing.text_normalization.es.graph_utils import ones
from fun_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

unit_plural_fem = pynini.string_file(get_abs_path("data/measures/measurements_plural_fem.tsv"))
unit_plural_masc = pynini.string_file(get_abs_path("data/measures/measurements_plural_masc.tsv"))

unit_singular_fem = pynini.project(unit_plural_fem, "input")
unit_singular_masc = pynini.project(unit_plural_masc, "input")

unit_plural_fem = pynini.project(unit_plural_fem, "output")
unit_plural_masc = pynini.project(unit_plural_masc, "output")


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, e.g.
        measure { cardinal { integer: "dos" units: "gramos" } } -> "dos gramos"
        measure { decimal { integer_part: "dos" quantity: "millones" units: "gramos" } } -> "dos millones de gramos"

    Args:
        decimal: DecimalFst
        cardinal: CardinalFst
        fraction: FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self, decimal: GraphFst, cardinal: GraphFst, fraction: GraphFst, deterministic: bool
    ):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        graph_decimal_masc = decimal.delete_tokens(decimal.graph_masc)
        graph_decimal_fem = decimal.delete_tokens(decimal.graph_fem)
        graph_cardinal_masc = cardinal.delete_tokens(cardinal.graph_masc)
        graph_cardinal_fem = cardinal.delete_tokens(cardinal.graph_fem)
        graph_fraction_fem = fraction.delete_tokens(fraction.graph_fem)
        graph_fraction_masc = fraction.delete_tokens(fraction.graph_masc)

        unit_masc = (unit_plural_masc | unit_singular_masc) + pynini.closure(
            DAMO_WHITE_SPACE + "por" + pynini.closure(DAMO_NOT_QUOTE, 1), 0, 1
        )
        unit_masc |= "por" + pynini.closure(DAMO_NOT_QUOTE, 1)
        unit_masc = (
            pynutil.delete('units: "')
            + (pynini.closure(DAMO_NOT_QUOTE) @ unit_masc)
            + pynutil.delete('"')
        )

        unit_fem = (unit_plural_fem | unit_singular_fem) + pynini.closure(
            DAMO_WHITE_SPACE + "por" + pynini.closure(DAMO_NOT_QUOTE, 1), 0, 1
        )
        unit_fem = (
            pynutil.delete('units: "')
            + (pynini.closure(DAMO_NOT_QUOTE) @ unit_fem)
            + pynutil.delete('"')
        )

        graph_masc = (graph_cardinal_masc | graph_decimal_masc) + DAMO_WHITE_SPACE + unit_masc
        graph_masc |= graph_fraction_masc + DAMO_WHITE_SPACE + pynutil.insert("de ") + unit_masc
        graph_masc |= pynutil.add_weight(
            graph_fraction_masc @ (DAMO_SIGMA + pynini.union("medio", "medios"))
            + DAMO_WHITE_SPACE
            + unit_masc,
            -0.001,
        )  # "medio litro" not "medio de litro"

        graph_fem = (graph_cardinal_fem | graph_decimal_fem) + DAMO_WHITE_SPACE + unit_fem
        graph_fem |= graph_fraction_fem + DAMO_WHITE_SPACE + pynutil.insert("de ") + unit_fem
        graph_fem |= pynutil.add_weight(
            graph_fraction_fem @ (DAMO_SIGMA + pynini.union("media", "medias"))
            + DAMO_WHITE_SPACE
            + unit_fem,
            -0.001,
        )

        graph = graph_masc | graph_fem

        graph = (
            pynini.cdrewrite(
                pynutil.insert(" de"),
                'quantity: "' + pynini.closure(DAMO_NOT_QUOTE, 1),
                '"',
                DAMO_SIGMA,
            )
            @ graph
        )  # billones de xyz

        graph @= pynini.cdrewrite(
            pynini.cross(ones, "uno"), "", DAMO_WHITE_SPACE + "por", DAMO_SIGMA
        )

        # To manage alphanumeric combonations ("a-8, 5x"), we let them use a weighted default path.
        alpha_num_unit = (
            pynutil.delete('units: "') + pynini.closure(DAMO_NOT_QUOTE) + pynutil.delete('"')
        )
        graph_alpha_num = pynini.union(
            (graph_cardinal_masc | graph_decimal_masc) + DAMO_SPACE + alpha_num_unit,
            alpha_num_unit + delete_extra_space + (graph_cardinal_masc | graph_decimal_masc),
        )

        graph |= pynutil.add_weight(graph_alpha_num, 0.01)

        graph += delete_preserve_order

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
