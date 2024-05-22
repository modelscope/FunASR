import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_SIGMA, GraphFst
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals. Numbers below ten are not converted.
    Allows both compound numeral strings or separated by whitespace.
    "und" (en: "and") can be inserted between "hundert" and following number or "tausend" and following single or double digit number.

        e.g. minus drei und zwanzig -> cardinal { negative: "-" integer: "23" } }
        e.g. minus dreiundzwanzig -> cardinal { integer: "23" } }
        e.g. dreizehn -> cardinal { integer: "13" } }
        e.g. ein hundert -> cardinal { integer: "100" } }
        e.g. einhundert -> cardinal { integer: "100" } }
        e.g. ein tausend -> cardinal { integer: "1000" } }
        e.g. eintausend -> cardinal { integer: "1000" } }
        e.g. ein tausend zwanzig -> cardinal { integer: "1020" } }

    Args:
        tn_cardinal_tagger: TN cardinal tagger
    """

    def __init__(self, tn_cardinal_tagger: GraphFst, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # add_space_between_chars = pynini.cdrewrite(pynini.closure(insert_space, 0, 1), DAMO_CHAR, DAMO_CHAR, DAMO_SIGMA)
        optional_delete_space = pynini.closure(DAMO_SIGMA | pynutil.delete(" "))

        graph = (tn_cardinal_tagger.graph @ optional_delete_space).invert().optimize()
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            (
                tn_cardinal_tagger.graph_hundred_component_at_least_one_none_zero_digit
                @ optional_delete_space
            )
            .invert()
            .optimize()
        )

        self.graph_ties = (
            (tn_cardinal_tagger.two_digit_non_zero @ optional_delete_space).invert().optimize()
        )
        # this is to make sure if there is an ambiguity with decimal, decimal is chosen, e.g. 1000000 vs. 1 million
        graph = pynutil.add_weight(graph, weight=0.001)
        self.graph_no_exception = graph
        self.digit = (
            pynini.arcmap(tn_cardinal_tagger.digit, map_type="rmweight").invert().optimize()
        )
        graph_exception = pynini.project(self.digit, "input")
        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        self.optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus ", '"-" '), 0, 1
        )

        final_graph = (
            self.optional_minus_graph
            + pynutil.insert('integer: "')
            + self.graph
            + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
