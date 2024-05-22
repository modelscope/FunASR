import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    GraphFst,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.ru.alphabet import RU_ALPHA_OR_SPACE
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone, which includes country code, number part and extension

    E.g
    "8-913-983-56-01" -> telephone { number_part: "восемь девятьсот тринадцать девятьсот восемьдесят три пятьдесят шесть ноль один" }

    Args:
        number_names: number_names for cardinal and ordinal numbers
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, number_names: dict, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        separator = pynini.cross("-", " ")  # between components
        number = number_names["cardinal_names_nominative"]

        country_code = (
            pynutil.insert('country_code: "')
            + pynini.closure(pynutil.add_weight(pynutil.delete("+"), 0.1), 0, 1)
            + number
            + separator
            + pynutil.insert('"')
        )
        optional_country_code = pynini.closure(country_code + insert_space, 0, 1)

        number_part = (
            DAMO_DIGIT**3 @ number
            + separator
            + DAMO_DIGIT**3 @ number
            + separator
            + DAMO_DIGIT**2 @ number
            + separator
            + DAMO_DIGIT**2 @ (pynini.closure(pynini.cross("0", "ноль ")) + number)
        )
        number_part = pynutil.insert('number_part: "') + number_part + pynutil.insert('"')
        tagger_graph = (optional_country_code + number_part).optimize()

        # verbalizer
        verbalizer_graph = pynini.closure(
            pynutil.delete('country_code: "')
            + pynini.closure(RU_ALPHA_OR_SPACE, 1)
            + pynutil.delete('"')
            + delete_space,
            0,
            1,
        )
        verbalizer_graph += (
            pynutil.delete('number_part: "')
            + pynini.closure(RU_ALPHA_OR_SPACE, 1)
            + pynutil.delete('"')
        )
        verbalizer_graph = verbalizer_graph.optimize()

        self.final_graph = (tagger_graph @ verbalizer_graph).optimize()
        self.fst = self.add_tokens(
            pynutil.insert('number_part: "') + self.final_graph + pynutil.insert('"')
        ).optimize()
