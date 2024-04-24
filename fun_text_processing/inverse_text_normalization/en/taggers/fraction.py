from fun_text_processing.text_normalization.en.graph_utils import GraphFst


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    """

    def __init__(self):
        super().__init__(name="fraction", kind="classify")
        # integer_part # numerator # denominator
