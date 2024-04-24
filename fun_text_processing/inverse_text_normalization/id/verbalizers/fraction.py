from fun_text_processing.inverse_text_normalization.id.graph_utils import GraphFst


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction,
    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")
