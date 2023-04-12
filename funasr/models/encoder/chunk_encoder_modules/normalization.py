"""Normalization modules for X-former blocks."""

from typing import Dict, Optional, Tuple

import torch


def get_normalization(
    normalization_type: str,
    eps: Optional[float] = None,
    partial: Optional[float] = None,
) -> Tuple[torch.nn.Module, Dict]:
    """Get normalization module and arguments given parameters.

    Args:
        normalization_type: Normalization module type.
        eps: Value added to the denominator.
        partial: Value defining the part of the input used for RMS stats (RMSNorm).

    Return:
        : Normalization module class
        : Normalization module arguments

    """
    norm = {
        "basic_norm": (
            BasicNorm,
            {"eps": eps if eps is not None else 0.25},
        ),
        "layer_norm": (torch.nn.LayerNorm, {"eps": eps if eps is not None else 1e-12}),
        "rms_norm": (
            RMSNorm,
            {
                "eps": eps if eps is not None else 1e-05,
                "partial": partial if partial is not None else -1.0,
            },
        ),
        "scale_norm": (
            ScaleNorm,
            {"eps": eps if eps is not None else 1e-05},
        ),
    }

    return norm[normalization_type]


class BasicNorm(torch.nn.Module):
    """BasicNorm module definition.

    Reference: https://github.com/k2-fsa/icefall/pull/288

    Args:
        normalized_shape: Expected size.
        eps: Value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 0.25,
    ) -> None:
        """Construct a BasicNorm object."""
        super().__init__()

        self.eps = torch.nn.Parameter(torch.tensor(eps).log().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute basic normalization.

        Args:
            x: Input sequences. (B, T, D_hidden)

        Returns:
            : Output sequences. (B, T, D_hidden)

        """
        scales = (torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps.exp()) ** -0.5

        return x * scales


class RMSNorm(torch.nn.Module):
    """RMSNorm module definition.

    Reference: https://arxiv.org/pdf/1910.07467.pdf

    Args:
        normalized_shape: Expected size.
        eps: Value added to the denominator for numerical stability.
        partial: Value defining the part of the input used for RMS stats.

    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        partial: float = 0.0,
    ) -> None:
        """Construct a RMSNorm object."""
        super().__init__()

        self.normalized_shape = normalized_shape

        self.partial = True if 0 < partial < 1 else False
        self.p = partial
        self.eps = eps

        self.scale = torch.nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization.

        Args:
            x: Input sequences. (B, T, D_hidden)

        Returns:
            x: Output sequences. (B, T, D_hidden)

        """
        if self.partial:
            partial_size = int(self.normalized_shape * self.p)
            partial_x, _ = torch.split(
                x, [partial_size, self.normalized_shape - partial_size], dim=-1
            )

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        else:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.normalized_shape

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x = self.scale * (x / (rms_x + self.eps))

        return x


class ScaleNorm(torch.nn.Module):
    """ScaleNorm module definition.

    Reference: https://arxiv.org/pdf/1910.05895.pdf

    Args:
        normalized_shape: Expected size.
        eps: Value added to the denominator for numerical stability.

    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        """Construct a ScaleNorm object."""
        super().__init__()

        self.eps = eps
        self.scale = torch.nn.Parameter(torch.tensor(normalized_shape**0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scale normalization.

        Args:
            x: Input sequences. (B, T, D_hidden)

        Returns:
            : Output sequences. (B, T, D_hidden)

        """
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)

        return x * norm
