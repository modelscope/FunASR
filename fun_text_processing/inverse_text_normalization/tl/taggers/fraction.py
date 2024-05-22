from fun_text_processing.inverse_text_normalization.tl.graph_utils import GraphFst


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    """

    def __init__(self):
        super().__init__(name="fraction", kind="classify")
        # integer_part # numerator # denominator
