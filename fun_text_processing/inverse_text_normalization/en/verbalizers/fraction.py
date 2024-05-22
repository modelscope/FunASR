from fun_text_processing.text_normalization.en.graph_utils import GraphFst


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction,
    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")
