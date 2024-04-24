import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NON_BREAKING_SPACE,
    DAMO_NOT_QUOTE,
    DAMO_SPACE,
    GraphFst,
    insert_space,
)
from fun_text_processing.text_normalization.ru.alphabet import RU_ALPHA
from fun_text_processing.text_normalization.ru.utils import get_abs_path
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure,  e.g.
        "2 кг" -> measure { cardinal { integer: "два килограма" } }
        This class also converts words containing numbers and letters
        e.g. "тест-8" —> "тест восемь"
        e.g. "тест-1,02" —> "тест одна целая две сотых"

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        # adding weight to make sure the space is preserved for ITN
        delete_space = pynini.closure(
            pynutil.add_weight(
                pynutil.delete(pynini.union(DAMO_SPACE, DAMO_NON_BREAKING_SPACE)), -1
            ),
            0,
            1,
        )

        cardinal_graph = cardinal.cardinal_numbers_default
        cardinal_graph_nominative = cardinal.cardinal_numbers_nominative
        graph_unit = pynini.string_file(get_abs_path("data/measurements.tsv"))
        optional_graph_negative = cardinal.optional_graph_negative

        space_for_units = (
            pynutil.add_weight(pynutil.insert(DAMO_NON_BREAKING_SPACE), -0.1)
            | pynutil.add_weight(pynutil.insert(DAMO_SPACE), 0.1)
        ).optimize()
        slash_unit = (
            (pynini.cross("/", "в") | pynini.cross("/", "за")) + space_for_units + graph_unit
        )

        unit_slash_unit = pynutil.add_weight(graph_unit + space_for_units + slash_unit, -0.1)
        default_units = (
            pynutil.insert('units: "') + (graph_unit | unit_slash_unit) + pynutil.insert('"')
        )
        slash_units = pynutil.insert('units: "') + slash_unit + pynutil.insert('"')
        subgraph_decimal = decimal.final_graph + ((delete_space + default_units) | slash_units)

        cardinal_space = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + cardinal_graph
            + (
                (delete_space + pynutil.insert('"') + pynutil.insert(" } ") + default_units)
                | (pynutil.insert('"') + pynutil.insert(" } ") + slash_units)
            )
        )

        cardinal_optional_dash_alpha = (
            pynutil.insert('cardinal { integer: "')
            + cardinal_graph
            + pynini.closure(pynini.cross("-", ""), 0, 1)
            + pynutil.insert('" } units: "')
            + pynini.closure(RU_ALPHA, 1)
            + pynutil.insert('"')
        )

        alpha_optional_dash_cardinal = (
            pynutil.insert('units: "')
            + pynini.closure(RU_ALPHA, 1)
            + pynini.closure(pynini.cross("-", ""), 0, 1)
            + pynutil.insert('"')
            + pynutil.insert(' cardinal { integer: "')
            + cardinal_graph_nominative
            + pynutil.insert('" } preserve_order: true')
        )

        decimal_dash_alpha = (
            decimal.final_graph
            + pynini.cross("-", "")
            + pynutil.insert(' units: "')
            + pynini.closure(RU_ALPHA, 1)
            + pynutil.insert('"')
        )

        alpha_dash_decimal = (
            pynutil.insert('units: "')
            + pynini.closure(RU_ALPHA, 1)
            + pynini.cross("-", "")
            + pynutil.insert('" ')
            + decimal.final_graph
            + pynutil.insert(" preserve_order: true")
        )

        self.tagger_graph_default = (subgraph_decimal | cardinal_space).optimize()

        tagger_graph = (
            self.tagger_graph_default
            | cardinal_optional_dash_alpha
            | alpha_optional_dash_cardinal
            | decimal_dash_alpha
            | alpha_dash_decimal
        ).optimize()

        # verbalizer
        unit = (
            pynutil.delete('units: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + delete_space
        )

        optional_sign = pynini.closure(pynini.cross('negative: "true" ', "минус "), 0, 1)
        integer = pynutil.delete(' "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        integer_part = pynutil.delete("integer_part:") + integer
        fractional_part = pynutil.delete("fractional_part:") + integer
        optional_quantity_part = pynini.closure(
            pynini.accep(" ")
            + pynutil.delete('quantity: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"'),
            0,
            1,
        )
        graph_decimal = (
            optional_sign
            + integer_part
            + pynini.accep(" ")
            + fractional_part
            + optional_quantity_part
        )

        graph_decimal = (
            pynutil.delete("decimal {")
            + delete_space
            + graph_decimal
            + delete_space
            + pynutil.delete("}")
        )

        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + optional_sign
            + pynutil.delete('integer: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete("}")
        )

        verbalizer_graph = (graph_cardinal | graph_decimal) + delete_space + insert_space + unit

        # SH adds "preserve_order: true" by default
        preserve_order = (
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
        )
        verbalizer_graph |= (
            unit
            + insert_space
            + (graph_cardinal | graph_decimal)
            + delete_space
            + pynini.closure(preserve_order, 0, 1)
        )
        self.verbalizer_graph = verbalizer_graph.optimize()

        final_graph = (tagger_graph @ verbalizer_graph).optimize()
        self.fst = self.add_tokens(
            pynutil.insert('cardinal { integer: "') + final_graph + pynutil.insert('" }')
        ).optimize()
