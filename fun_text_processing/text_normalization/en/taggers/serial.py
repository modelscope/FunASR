import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_NOT_SPACE,
    DAMO_SIGMA,
    GraphFst,
    convert_space,
)
from fun_text_processing.text_normalization.en.utils import get_abs_path, load_labels
from pynini.examples import plurals
from pynini.lib import pynutil


class SerialFst(GraphFst):
    """
    This class is a composite class of two other class instances

    Args:
        time: composed tagger and verbalizer
        date: composed tagger and verbalizer
        cardinal: tagger
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
        lm: whether to use for hybrid LM
    """

    def __init__(
        self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True, lm: bool = False
    ):
        super().__init__(name="integer", kind="classify", deterministic=deterministic)

        """
        Finite state transducer for classifying serial (handles only cases without delimiters,
        values with delimiters are handled by default).
            The serial is a combination of digits, letters and dashes, e.g.:
            c325b -> tokens { cardinal { integer: "c three two five b" } }
        """
        num_graph = pynini.compose(DAMO_DIGIT ** (6, ...), cardinal.single_digits_graph).optimize()
        num_graph |= pynini.compose(DAMO_DIGIT ** (1, 5), cardinal.graph).optimize()
        # to handle numbers starting with zero
        num_graph |= pynini.compose(
            pynini.accep("0") + pynini.closure(DAMO_DIGIT), cardinal.single_digits_graph
        ).optimize()
        # TODO: "#" doesn't work from the file
        symbols_graph = pynini.string_file(
            get_abs_path("data/whitelist/symbol.tsv")
        ).optimize() | pynini.cross("#", "hash")
        num_graph |= symbols_graph

        if not self.deterministic and not lm:
            num_graph |= cardinal.single_digits_graph
            # also allow double digits to be pronounced as integer in serial number
            num_graph |= pynutil.add_weight(
                DAMO_DIGIT**2 @ cardinal.graph_hundred_component_at_least_one_none_zero_digit,
                weight=0.0001,
            )

        # add space between letter and digit/symbol
        symbols = [x[0] for x in load_labels(get_abs_path("data/whitelist/symbol.tsv"))]
        symbols = pynini.union(*symbols)
        digit_symbol = DAMO_DIGIT | symbols

        graph_with_space = pynini.compose(
            pynini.cdrewrite(pynutil.insert(" "), DAMO_ALPHA | symbols, digit_symbol, DAMO_SIGMA),
            pynini.cdrewrite(pynutil.insert(" "), digit_symbol, DAMO_ALPHA | symbols, DAMO_SIGMA),
        )

        # serial graph with delimiter
        delimiter = pynini.accep("-") | pynini.accep("/") | pynini.accep(" ")
        if not deterministic:
            delimiter |= pynini.cross("-", " dash ") | pynini.cross("/", " slash ")

        alphas = pynini.closure(DAMO_ALPHA, 1)
        letter_num = alphas + delimiter + num_graph
        num_letter = pynini.closure(num_graph + delimiter, 1) + alphas
        next_alpha_or_num = pynini.closure(delimiter + (alphas | num_graph))
        next_alpha_or_num |= pynini.closure(
            delimiter
            + num_graph
            + plurals._priority_union(pynini.accep(" "), pynutil.insert(" "), DAMO_SIGMA).optimize()
            + alphas
        )

        serial_graph = letter_num + next_alpha_or_num
        serial_graph |= num_letter + next_alpha_or_num
        # numbers only with 2+ delimiters
        serial_graph |= (
            num_graph
            + delimiter
            + num_graph
            + delimiter
            + num_graph
            + pynini.closure(delimiter + num_graph)
        )
        # 2+ symbols
        serial_graph |= pynini.compose(
            DAMO_SIGMA + symbols + DAMO_SIGMA, num_graph + delimiter + num_graph
        )

        # exclude ordinal numbers from serial options
        serial_graph = pynini.compose(
            pynini.difference(DAMO_SIGMA, pynini.project(ordinal.graph, "input")), serial_graph
        ).optimize()

        serial_graph = pynutil.add_weight(serial_graph, 0.0001)
        serial_graph |= (
            pynini.closure(DAMO_NOT_SPACE, 1)
            + (pynini.cross("^2", " squared") | pynini.cross("^3", " cubed")).optimize()
        )

        # at least one serial graph with alpha numeric value and optional additional serial/num/alpha values
        serial_graph = (
            pynini.closure((serial_graph | num_graph | alphas) + delimiter)
            + serial_graph
            + pynini.closure(delimiter + (serial_graph | num_graph | alphas))
        )

        serial_graph |= pynini.compose(graph_with_space, serial_graph.optimize()).optimize()
        serial_graph = pynini.compose(pynini.closure(DAMO_NOT_SPACE, 2), serial_graph).optimize()

        # this is not to verbolize "/" as "slash" in cases like "import/export"
        serial_graph = pynini.compose(
            pynini.difference(
                DAMO_SIGMA,
                pynini.closure(DAMO_ALPHA, 1) + pynini.accep("/") + pynini.closure(DAMO_ALPHA, 1),
            ),
            serial_graph,
        )
        self.graph = serial_graph.optimize()
        graph = (
            pynutil.insert('name: "') + convert_space(self.graph).optimize() + pynutil.insert('"')
        )
        self.fst = graph.optimize()
