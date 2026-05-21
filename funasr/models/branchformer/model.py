import logging

from funasr.models.transformer.model import Transformer
from funasr.register import tables


@tables.register("model_classes", "Branchformer")
class Branchformer(Transformer):
    """Branchformer: Parallel branch encoder architecture.

    Uses parallel branches of self-attention and convolution that are
    merged via concatenation. Alternative to Conformer with similar accuracy.

    Inherits Transformer pipeline for training and inference.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        """Initialize Branchformer.
        
            Args:
                *args: Variable positional arguments.
                **kwargs: Additional keyword arguments.
            """
        super().__init__(*args, **kwargs)
