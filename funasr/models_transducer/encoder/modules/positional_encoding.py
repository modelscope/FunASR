"""Positional encoding modules."""

import math

import torch

from funasr.modules.embedding import _pre_hook


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding.

    Args:
        size: Module size.
        max_len: Maximum input length.
        dropout_rate: Dropout rate.

    """

    def __init__(
        self, size: int, dropout_rate: float = 0.0, max_len: int = 5000
    ) -> None:
        """Construct a RelativePositionalEncoding object."""
        super().__init__()

        self.size = size

        self.pe = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x: torch.Tensor, left_context: int = 0) -> None:
        """Reset positional encoding.

        Args:
            x: Input sequences. (B, T, ?)
            left_context: Number of frames in left context.

        """
        time1 = x.size(1) + left_context

        if self.pe is not None:
            if self.pe.size(1) >= time1 * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(device=x.device, dtype=x.dtype)
                return

        pe_positive = torch.zeros(time1, self.size)
        pe_negative = torch.zeros(time1, self.size)

        position = torch.arange(0, time1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.size, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.size)
        )

        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)

        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_negative = pe_negative[1:].unsqueeze(0)

        self.pe = torch.cat([pe_positive, pe_negative], dim=1).to(
            dtype=x.dtype, device=x.device
        )

    def forward(self, x: torch.Tensor, left_context: int = 0) -> torch.Tensor:
        """Compute positional encoding.

        Args:
            x: Input sequences. (B, T, ?)
            left_context: Number of frames in left context.

        Returns:
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), ?)

        """
        self.extend_pe(x, left_context=left_context)

        time1 = x.size(1) + left_context

        pos_enc = self.pe[
            :, self.pe.size(1) // 2 - time1 + 1 : self.pe.size(1) // 2 + x.size(1)
        ]
        pos_enc = self.dropout(pos_enc)

        return pos_enc
