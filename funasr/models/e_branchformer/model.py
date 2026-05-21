import logging

from funasr.models.transformer.model import Transformer
from funasr.register import tables


@tables.register("model_classes", "EBranchformer")
class EBranchformer(Transformer):
    """E-Branchformer: Enhanced Branchformer with improved merging.

    Uses element-wise merging instead of concatenation for parallel branches,
    resulting in better parameter efficiency.

    Inherits Transformer pipeline.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        """Initialize EBranchformer.
        
            Args:
                *args: Variable positional arguments.
                **kwargs: Additional keyword arguments.
            """
        super().__init__(*args, **kwargs)
