import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone numbers, e.g.
        telephone { country_code: "one" number_part: "one two three, one two three, five six seven eight" extension: "one"  }
        -> one, one two three, one two three, five six seven eight, one

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="verbalize", deterministic=deterministic)

        optional_country_code = pynini.closure(
            pynutil.delete('country_code: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + delete_space
            + insert_space,
            0,
            1,
        )

        number_part = (
            pynutil.delete('number_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynini.closure(pynutil.add_weight(pynutil.delete(" "), -0.0001), 0, 1)
            + pynutil.delete('"')
        )

        optional_extension = pynini.closure(
            delete_space
            + insert_space
            + pynutil.delete('extension: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"'),
            0,
            1,
        )

        graph = optional_country_code + number_part + optional_extension
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
