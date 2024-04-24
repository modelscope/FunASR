from collections import defaultdict

import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    DAMO_SPACE,
    GraphFst,
    insert_space,
)
from fun_text_processing.text_normalization.en.utils import load_labels
from fun_text_processing.text_normalization.ru.utils import get_abs_path
from pynini.lib import pynutil

delete_space = pynutil.delete(" ")


def prepare_labels_for_insertion(file_path: str):
    """
    Read the file and creates a union insertion graph

    Args:
        file_path: path to a file (3 columns: a label type e.g.
        "@@decimal_delimiter@@", a label e.g. "целого", and a weight e.g. "0.1").

    Returns dictionary mapping from label type to an fst that inserts the labels with the specified weights.

    """
    labels = load_labels(file_path)
    mapping = defaultdict(list)
    for k, v, w in labels:
        mapping[k].append((v, w))

    for k in mapping:
        mapping[k] = (
            insert_space
            + pynini.union(
                *[pynutil.add_weight(pynutil.insert(end), weight) for end, weight in mapping[k]]
            )
        ).optimize()
    return mapping


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        "1,08" -> tokens { decimal { integer_part: "одно целая" fractional_part: "восемь сотых} }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = False):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        integer_part = cardinal.cardinal_numbers_default
        cardinal_numbers_with_leading_zeros = cardinal.cardinal_numbers_with_leading_zeros

        delimiter_map = prepare_labels_for_insertion(
            get_abs_path("data/numbers/decimal_delimiter.tsv")
        )
        delimiter = (
            pynini.cross(",", "")
            + delimiter_map["@@decimal_delimiter@@"]
            + pynini.closure(pynutil.add_weight(pynutil.insert(" и"), 0.5), 0, 1)
        ).optimize()

        decimal_endings_map = prepare_labels_for_insertion(
            get_abs_path("data/numbers/decimal_endings.tsv")
        )

        self.integer_part = integer_part + delimiter
        graph_integer = pynutil.insert('integer_part: "') + self.integer_part + pynutil.insert('"')

        graph_fractional = (
            DAMO_DIGIT @ cardinal_numbers_with_leading_zeros + decimal_endings_map["10"]
        )
        graph_fractional |= (
            DAMO_DIGIT + DAMO_DIGIT
        ) @ cardinal_numbers_with_leading_zeros + decimal_endings_map["100"]
        graph_fractional |= (
            DAMO_DIGIT + DAMO_DIGIT + DAMO_DIGIT
        ) @ cardinal_numbers_with_leading_zeros + decimal_endings_map["1000"]
        graph_fractional |= (
            DAMO_DIGIT + DAMO_DIGIT + DAMO_DIGIT + DAMO_DIGIT
        ) @ cardinal_numbers_with_leading_zeros + decimal_endings_map["10000"]

        self.optional_quantity = pynini.string_file(
            get_abs_path("data/numbers/quantity.tsv")
        ).optimize()

        self.graph_fractional = graph_fractional
        graph_fractional = (
            pynutil.insert('fractional_part: "') + graph_fractional + pynutil.insert('"')
        )
        optional_quantity = pynini.closure(
            (pynutil.add_weight(pynini.accep(DAMO_SPACE), -0.1) | insert_space)
            + pynutil.insert('quantity: "')
            + self.optional_quantity
            + pynutil.insert('"'),
            0,
            1,
        )
        self.final_graph = (
            cardinal.optional_graph_negative
            + graph_integer
            + insert_space
            + graph_fractional
            + optional_quantity
        )

        self.final_graph = self.add_tokens(self.final_graph)
        self.fst = self.final_graph.optimize()
