import pynini
from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        восемнадцатое июня две тысячи второго -> tokens { date { day: "18.06.2002" } }

    Args:
        tn_date: Text normalization Date graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_date: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        graph = pynini.invert(tn_date.final_graph).optimize()
        graph = self.add_tokens(pynutil.insert('day: "') + graph + pynutil.insert('"'))
        self.fst = graph.optimize()
