import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_NON_BREAKING_SPACE,
    DAMO_SIGMA,
    DAMO_SPACE,
    DAMO_UPPER,
    SINGULAR_TO_PLURAL,
    TO_LOWER,
    GraphFst,
    convert_space,
    delete_space,
    delete_zero_or_one_space,
    insert_space,
)
from fun_text_processing.text_normalization.en.taggers.ordinal import OrdinalFst as OrdinalTagger
from fun_text_processing.text_normalization.en.taggers.whitelist import get_formats
from fun_text_processing.text_normalization.en.utils import get_abs_path, load_labels
from fun_text_processing.text_normalization.en.verbalizers.ordinal import (
    OrdinalFst as OrdinalVerbalizer,
)
from pynini.examples import plurals
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g.
        -12kg -> measure { negative: "true" cardinal { integer: "twelve" } units: "kilograms" }
        1kg -> measure { cardinal { integer: "one" } units: "kilogram" }
        .5kg -> measure { decimal { fractional_part: "five" } units: "kilograms" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph_with_and | self.get_range(cardinal.graph_with_and)

        graph_unit = pynini.string_file(get_abs_path("data/measure/unit.tsv"))
        if not deterministic:
            graph_unit |= pynini.string_file(get_abs_path("data/measure/unit_alternatives.tsv"))

        graph_unit |= pynini.compose(
            pynini.closure(TO_LOWER, 1)
            + (DAMO_ALPHA | TO_LOWER)
            + pynini.closure(DAMO_ALPHA | TO_LOWER),
            graph_unit,
        ).optimize()

        graph_unit_plural = convert_space(graph_unit @ SINGULAR_TO_PLURAL)
        graph_unit = convert_space(graph_unit)

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1
        )

        graph_unit2 = (
            pynini.cross("/", "per")
            + delete_zero_or_one_space
            + pynutil.insert(DAMO_NON_BREAKING_SPACE)
            + graph_unit
        )

        optional_graph_unit2 = pynini.closure(
            delete_zero_or_one_space + pynutil.insert(DAMO_NON_BREAKING_SPACE) + graph_unit2,
            0,
            1,
        )

        unit_plural = (
            pynutil.insert('units: "')
            + (graph_unit_plural + optional_graph_unit2 | graph_unit2)
            + pynutil.insert('"')
        )

        unit_singular = (
            pynutil.insert('units: "')
            + (graph_unit + optional_graph_unit2 | graph_unit2)
            + pynutil.insert('"')
        )

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + delete_space
            + pynutil.insert(" } ")
            + unit_plural
        )

        # support radio FM/AM
        subgraph_decimal |= (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + delete_space
            + pynutil.insert(" } ")
            + pynutil.insert('units: "')
            + pynini.union("AM", "FM")
            + pynutil.insert('"')
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + ((DAMO_SIGMA - "1") @ cardinal_graph)
            + delete_space
            + pynutil.insert('"')
            + pynutil.insert(" } ")
            + unit_plural
        )

        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + pynini.cross("1", "one")
            + delete_space
            + pynutil.insert('"')
            + pynutil.insert(" } ")
            + unit_singular
        )

        unit_graph = (
            pynutil.insert('cardinal { integer: "-" } units: "')
            + pynini.cross(pynini.union("/", "per"), "per")
            + delete_zero_or_one_space
            + pynutil.insert(DAMO_NON_BREAKING_SPACE)
            + graph_unit
            + pynutil.insert('" preserve_order: true')
        )

        decimal_dash_alpha = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynini.cross("-", "")
            + pynutil.insert(' } units: "')
            + pynini.closure(DAMO_ALPHA, 1)
            + pynutil.insert('"')
        )

        decimal_times = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(' } units: "')
            + pynini.cross(pynini.union("x", "X"), "x")
            + pynutil.insert('"')
        )

        alpha_dash_decimal = (
            pynutil.insert('units: "')
            + pynini.closure(DAMO_ALPHA, 1)
            + pynini.accep("-")
            + pynutil.insert('"')
            + pynutil.insert(" decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(" } preserve_order: true")
        )

        subgraph_fraction = (
            pynutil.insert("fraction { ")
            + fraction.graph
            + delete_space
            + pynutil.insert(" } ")
            + unit_plural
        )

        address = self.get_address_graph(cardinal)
        address = (
            pynutil.insert('units: "address" cardinal { integer: "')
            + address
            + pynutil.insert('" } preserve_order: true')
        )

        math_operations = pynini.string_file(get_abs_path("data/measure/math_operation.tsv"))
        delimiter = pynini.accep(" ") | pynutil.insert(" ")

        math = (
            (cardinal_graph | DAMO_ALPHA)
            + delimiter
            + math_operations
            + (delimiter | DAMO_ALPHA)
            + cardinal_graph
            + delimiter
            + pynini.cross("=", "equals")
            + delimiter
            + (cardinal_graph | DAMO_ALPHA)
        )

        math |= (
            (cardinal_graph | DAMO_ALPHA)
            + delimiter
            + pynini.cross("=", "equals")
            + delimiter
            + (cardinal_graph | DAMO_ALPHA)
            + delimiter
            + math_operations
            + delimiter
            + cardinal_graph
        )

        math = (
            pynutil.insert('units: "math" cardinal { integer: "')
            + math
            + pynutil.insert('" } preserve_order: true')
        )
        final_graph = (
            subgraph_decimal
            | subgraph_cardinal
            | unit_graph
            | decimal_dash_alpha
            | decimal_times
            | alpha_dash_decimal
            | subgraph_fraction
            | address
            | math
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def get_range(self, cardinal: GraphFst):
        """
        Returns range forms for measure tagger, e.g. 2-3, 2x3, 2*2

        Args:
            cardinal: cardinal GraphFst
        """
        range_graph = cardinal + pynini.cross(pynini.union("-", " - "), " to ") + cardinal

        for x in [" x ", "x"]:
            range_graph |= cardinal + pynini.cross(x, " by ") + cardinal
            if not self.deterministic:
                range_graph |= cardinal + pynini.cross(x, " times ") + cardinal

        for x in ["*", " * "]:
            range_graph |= cardinal + pynini.cross(x, " times ") + cardinal
        return range_graph.optimize()

    def get_address_graph(self, cardinal):
        """
        Finite state transducer for classifying serial.
            The serial is a combination of digits, letters and dashes, e.g.:
            2788 San Tomas Expy, Santa Clara, CA 95051 ->
                units: "address" cardinal
                { integer: "two seven eight eight San Tomas Expressway Santa Clara California nine five zero five one" }
                 preserve_order: true
        """
        ordinal_verbalizer = OrdinalVerbalizer().graph
        ordinal_tagger = OrdinalTagger(cardinal=cardinal).graph
        ordinal_num = pynini.compose(
            pynutil.insert('integer: "') + ordinal_tagger + pynutil.insert('"'), ordinal_verbalizer
        )

        address_num = (
            DAMO_DIGIT ** (1, 2) @ cardinal.graph_hundred_component_at_least_one_none_zero_digit
        )
        address_num += insert_space + DAMO_DIGIT**2 @ (
            pynini.closure(pynini.cross("0", "zero "), 0, 1)
            + cardinal.graph_hundred_component_at_least_one_none_zero_digit
        )
        # to handle the rest of the numbers
        address_num = pynini.compose(DAMO_DIGIT ** (3, 4), address_num)
        address_num = plurals._priority_union(address_num, cardinal.graph, DAMO_SIGMA)

        direction = (
            pynini.cross("E", "East")
            | pynini.cross("S", "South")
            | pynini.cross("W", "West")
            | pynini.cross("N", "North")
        ) + pynini.closure(pynutil.delete("."), 0, 1)

        direction = pynini.closure(pynini.accep(DAMO_SPACE) + direction, 0, 1)
        address_words = get_formats(get_abs_path("data/address/address_word.tsv"))
        address_words = (
            pynini.accep(DAMO_SPACE)
            + (pynini.closure(ordinal_num, 0, 1) | DAMO_UPPER + pynini.closure(DAMO_ALPHA, 1))
            + DAMO_SPACE
            + pynini.closure(DAMO_UPPER + pynini.closure(DAMO_ALPHA) + DAMO_SPACE)
            + address_words
        )

        city = pynini.closure(DAMO_ALPHA | pynini.accep(DAMO_SPACE), 1)
        city = pynini.closure(pynini.accep(",") + pynini.accep(DAMO_SPACE) + city, 0, 1)

        states = load_labels(get_abs_path("data/address/state.tsv"))

        additional_options = []
        for x, y in states:
            additional_options.append((x, f"{y[0]}.{y[1:]}"))
        states.extend(additional_options)
        state_graph = pynini.string_map(states)
        state = pynini.invert(state_graph)
        state = pynini.closure(pynini.accep(",") + pynini.accep(DAMO_SPACE) + state, 0, 1)

        zip_code = pynini.compose(DAMO_DIGIT**5, cardinal.single_digits_graph)
        zip_code = pynini.closure(
            pynini.closure(pynini.accep(","), 0, 1) + pynini.accep(DAMO_SPACE) + zip_code,
            0,
            1,
        )

        address = (
            address_num + direction + address_words + pynini.closure(city + state + zip_code, 0, 1)
        )

        address |= (
            address_num + direction + address_words + pynini.closure(pynini.cross(".", ""), 0, 1)
        )

        return address
