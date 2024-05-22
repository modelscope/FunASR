import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_SPACE, GraphFst
from fun_text_processing.text_normalization.ru.verbalizers.time import TimeFst as TNTimeVerbalizer
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        "два часа пятнадцать минут" -> time { hours: "02:15" }

    Args:
        tn_time: Text Normalization Time graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_time: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        tn_time_tagger = tn_time.graph_preserve_order
        tn_time_verbalizer = TNTimeVerbalizer().graph
        tn_time_graph_preserve_order = pynini.compose(tn_time_tagger, tn_time_verbalizer).optimize()
        graph_preserve_order = pynini.invert(tn_time_graph_preserve_order).optimize()
        graph_preserve_order = (
            pynutil.insert('hours: "') + graph_preserve_order + pynutil.insert('"')
        )

        # "пятнадцать минут шестого" -> 17:15
        # Requires permutations for the correct verbalization
        m_next_h = (
            pynutil.insert('minutes: "')
            + pynini.invert(tn_time.minutes).optimize()
            + pynutil.insert('"')
            + pynini.accep(DAMO_SPACE)
            + pynutil.insert('hours: "')
            + pynini.invert(tn_time.increment_hour_ordinal).optimize()
            + pynutil.insert('"')
        ).optimize()

        # "без пятнадцати минут шесть" -> 17:45
        # Requires permutation for the correct verbalization
        m_to_h = (
            pynini.cross("без ", 'minutes: "')
            + pynini.invert(tn_time.mins_to_h)
            + pynutil.insert('"')
            + pynini.accep(DAMO_SPACE)
            + pynutil.insert('hours: "')
            + pynini.invert(tn_time.increment_hour_cardinal).optimize()
            + pynutil.insert('"')
        )

        graph_reserve_order = m_next_h | m_to_h
        graph = graph_preserve_order | graph_reserve_order
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
