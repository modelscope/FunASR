import logging

import torch

from funasr.models.transformer.model import Transformer
from funasr.register import tables


@tables.register("model_classes", "Conformer")
class Conformer(Transformer):
    """Conformer: CTC-attention hybrid encoder-decoder model.

    Combines convolution and self-attention in the encoder for better
    local and global context modeling. Inherits full Transformer pipeline
    (CTC + attention decoder + beam search).

    Output: {"key": str, "text": str}
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        """Initialize Conformer.
        
            Args:
                *args: Variable positional arguments.
                **kwargs: Additional keyword arguments.
            """
        super().__init__(*args, **kwargs)
