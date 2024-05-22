import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    GraphFst,
    convert_space,
)
from pynini.lib import pynutil


class RangeFst(GraphFst):
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
        self,
        time: GraphFst,
        date: GraphFst,
        cardinal: GraphFst,
        deterministic: bool = True,
        lm: bool = False,
    ):
        super().__init__(name="range", kind="classify", deterministic=deterministic)

        delete_space = pynini.closure(pynutil.delete(" "), 0, 1)

        approx = pynini.cross("~", "approximately")

        # TIME
        time_graph = time + delete_space + pynini.cross("-", " to ") + delete_space + time
        self.graph = time_graph | (approx + time)

        cardinal = cardinal.graph_with_and
        # YEAR
        date_year_four_digit = (DAMO_DIGIT**4 + pynini.closure(pynini.accep("s"), 0, 1)) @ date
        date_year_two_digit = (DAMO_DIGIT**2 + pynini.closure(pynini.accep("s"), 0, 1)) @ date
        year_to_year_graph = (
            date_year_four_digit
            + delete_space
            + pynini.cross("-", " to ")
            + delete_space
            + (date_year_four_digit | date_year_two_digit | (DAMO_DIGIT**2 @ cardinal))
        )
        mid_year_graph = (
            pynini.accep("mid")
            + pynini.cross("-", " ")
            + (date_year_four_digit | date_year_two_digit)
        )

        self.graph |= year_to_year_graph
        self.graph |= mid_year_graph

        # ADDITION
        range_graph = cardinal + pynini.closure(pynini.cross("+", " plus ") + cardinal, 1)
        range_graph |= cardinal + pynini.closure(pynini.cross(" + ", " plus ") + cardinal, 1)
        range_graph |= approx + cardinal
        range_graph |= cardinal + (pynini.cross("...", " ... ") | pynini.accep(" ... ")) + cardinal

        if not deterministic or lm:
            # cardinal ----
            cardinal_to_cardinal_graph = (
                cardinal
                + delete_space
                + pynini.cross("-", pynini.union(" to ", " minus "))
                + delete_space
                + cardinal
            )

            range_graph |= cardinal_to_cardinal_graph | (
                cardinal + delete_space + pynini.cross(":", " to ") + delete_space + cardinal
            )

            # MULTIPLY
            for x in [" x ", "x"]:
                range_graph |= cardinal + pynini.closure(
                    pynini.cross(x, pynini.union(" by ", " times ")) + cardinal, 1
                )

            for x in ["*", " * "]:
                range_graph |= cardinal + pynini.closure(pynini.cross(x, " times ") + cardinal, 1)

            # supports "No. 12" -> "Number 12"
            range_graph |= (
                (pynini.cross(pynini.union("NO", "No"), "Number") | pynini.cross("no", "number"))
                + pynini.closure(pynini.union(". ", " "), 0, 1)
                + cardinal
            )

            for x in ["/", " / "]:
                range_graph |= cardinal + pynini.closure(
                    pynini.cross(x, " divided by ") + cardinal, 1
                )

        self.graph |= range_graph

        self.graph = self.graph.optimize()
        graph = (
            pynutil.insert('name: "') + convert_space(self.graph).optimize() + pynutil.insert('"')
        )
        self.fst = graph.optimize()
