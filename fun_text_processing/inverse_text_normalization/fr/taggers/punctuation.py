import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import GraphFst
from pynini.lib import pynutil


class PunctuationFst(GraphFst):
    """
    Finite state transducer for classifying punctuation
        e.g. a, -> tokens { name: "a" } tokens { name: "," }
    """

    def __init__(self):
        super().__init__(name="punctuation", kind="classify")

        s = "!#$%&'()*+,-./:;<=>?@^_`{|}~"
        guillemets = "\u00AB" + "\u00BB"  # quotation marks in French.
        s += guillemets
        punct = pynini.union(*s)

        graph = pynutil.insert('name: "') + punct + pynutil.insert('"')

        self.fst = graph.optimize()
