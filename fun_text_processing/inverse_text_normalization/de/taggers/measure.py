import pynini
from fun_text_processing.text_normalization.de.taggers.measure import (
    singular_to_plural,
    unit_singular,
)
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_SIGMA,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure. Allows for plural form for unit.
        e.g. minus elf kilogramm -> measure { cardinal { negative: "true" integer: "11" } units: "kg" }
        e.g. drei stunden -> measure { cardinal { integer: "3" } units: "h" }
        e.g. ein halb kilogramm -> measure { decimal { integer_part: "1/2" } units: "kg" }
        e.g. eins komma zwei kilogramm -> measure { decimal { integer_part: "1" fractional_part: "2" } units: "kg" }

    Args:
        itn_cardinal_tagger: ITN Cardinal tagger
        itn_decimal_tagger: ITN Decimal tagger
        itn_fraction_tagger: ITN Fraction tagger
    """

    def __init__(
        self,
        itn_cardinal_tagger: GraphFst,
        itn_decimal_tagger: GraphFst,
        itn_fraction_tagger: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        cardinal_graph = (
            pynini.cdrewrite(
                pynini.cross(pynini.union("ein", "eine"), "eins"), "[BOS]", "[EOS]", DAMO_SIGMA
            )
            @ itn_cardinal_tagger.graph_no_exception
        )

        graph_unit_singular = pynini.invert(unit_singular)  # singular -> abbr
        unit = (
            pynini.invert(singular_to_plural()) @ graph_unit_singular
        ) | graph_unit_singular  # plural -> abbr
        unit = convert_space(unit)
        graph_unit_singular = convert_space(graph_unit_singular)

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", '"true"') + delete_extra_space,
            0,
            1,
        )

        unit_misc = pynutil.insert("/") + pynutil.delete("pro") + delete_space + graph_unit_singular

        unit = (
            pynutil.insert('units: "')
            + (unit | unit_misc | pynutil.add_weight(unit + delete_space + unit_misc, 0.01))
            + pynutil.insert('"')
        )

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + itn_decimal_tagger.final_graph_wo_negative
            + pynutil.insert(" }")
            + delete_extra_space
            + unit
        )

        subgraph_fraction = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + pynutil.insert('integer_part: "')
            + itn_fraction_tagger.graph
            + pynutil.insert('" }')
            + delete_extra_space
            + unit
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + cardinal_graph
            + pynutil.insert('"')
            + pynutil.insert(" }")
            + delete_extra_space
            + unit
        )
        final_graph = subgraph_cardinal | subgraph_decimal | subgraph_fraction
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
