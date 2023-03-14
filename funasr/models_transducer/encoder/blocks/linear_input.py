"""LinearInput block for Transducer encoder."""

from typing import Optional, Tuple, Union

import torch

class LinearInput(torch.nn.Module):
    """ConvInput module definition.

    Args:
        input_size: Input size.
        conv_size: Convolution size.
        subsampling_factor: Subsampling factor.
        vgg_like: Whether to use a VGG-like network.
        output_size: Block output dimension.

    """

    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        subsampling_factor: int = 1,
    ) -> None:
        """Construct a ConvInput object."""
        super().__init__()
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
            torch.nn.LayerNorm(output_size),
            torch.nn.Dropout(0.1),
        )
        self.subsampling_factor = subsampling_factor
        self.min_frame_length = 1

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x = self.embed(x)
        return x, mask

    def get_size_before_subsampling(self, size: int) -> int:
        """Return the original size before subsampling for a given size.

        Args:
            size: Number of frames after subsampling.

        Returns:
            : Number of frames before subsampling.

        """
        return size
