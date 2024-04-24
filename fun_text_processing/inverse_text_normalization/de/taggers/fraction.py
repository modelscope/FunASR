import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    GraphFst,
    convert_space,
    delete_space,
)
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        e.g. ein halb -> tokens { name: "1/2" }
        e.g. ein ein halb -> tokens { name: "1 1/2" }
        e.g. drei zwei ein hundertstel -> tokens { name: "3 2/100" }

    Args:
        itn_cardinal_tagger: ITN cardinal tagger
        tn_fraction_verbalizer: TN fraction verbalizer
    """

    def __init__(
        self,
        itn_cardinal_tagger: GraphFst,
        tn_fraction_verbalizer: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)
        tagger = tn_fraction_verbalizer.graph.invert().optimize()

        delete_optional_sign = pynini.closure(
            pynutil.delete("negative: ") + pynini.cross('"true" ', "-"), 0, 1
        )
        delete_integer_marker = (
            pynutil.delete('integer_part: "')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        ) @ itn_cardinal_tagger.graph_no_exception

        delete_numerator_marker = (
            pynutil.delete('numerator: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        ) @ itn_cardinal_tagger.graph_no_exception

        delete_denominator_marker = (
            pynutil.insert("/")
            + (
                pynutil.delete('denominator: "')
                + pynini.closure(DAMO_NOT_QUOTE, 1)
                + pynutil.delete('"')
            )
            @ itn_cardinal_tagger.graph_no_exception
        )

        graph = (
            pynini.closure(delete_integer_marker + pynini.accep(" "), 0, 1)
            + delete_numerator_marker
            + delete_space
            + delete_denominator_marker
        ).optimize()
        verbalizer = delete_optional_sign + graph

        self.graph = tagger @ verbalizer

        graph = pynutil.insert('name: "') + convert_space(self.graph) + pynutil.insert('"')
        self.fst = graph.optimize()
