"""Abstract decoder definition for Transducer models."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import torch


class AbsDecoder(torch.nn.Module, ABC):
    """Abstract decoder module."""

    @abstractmethod
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Encode source label sequences.

        Args:
            labels: Label ID sequences. (B, L)

        Returns:
            dec_out: Decoder output sequences. (B, T, D_dec)

        """
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        label: torch.Tensor,
        label_sequence: List[int],
        dec_state: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]]:
        """One-step forward hypothesis.

        Args:
            label: Previous label. (1, 1)
            label_sequence: Current label sequence.
            dec_state: Previous decoder hidden states.
                         ((N, 1, D_dec), (N, 1, D_dec) or None) or None

        Returns:
            dec_out: Decoder output sequence. (1, D_dec) or (1, D_emb)
            dec_state: Decoder hidden states.
                         ((N, 1, D_dec), (N, 1, D_dec) or None) or None

        """
        raise NotImplementedError

    @abstractmethod
    def batch_score(
        self,
        hyps: List[Any],
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.

        Returns:
            dec_out: Decoder output sequences. (B, D_dec) or (B, D_emb)
            states: Decoder hidden states.
                      ((N, B, D_dec), (N, B, D_dec) or None) or None

        """
        raise NotImplementedError

    @abstractmethod
    def set_device(self, device: torch.Tensor) -> None:
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        raise NotImplementedError

    @abstractmethod
    def init_state(
        self, batch_size: int
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.tensor]]]:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            : Initial decoder hidden states.
                ((N, B, D_dec), (N, B, D_dec) or None) or None

        """
        raise NotImplementedError

    @abstractmethod
    def select_state(
        self,
        states: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
        idx: int = 0,
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Get specified ID state from batch of states, if provided.

        Args:
            states: Decoder hidden states.
                      ((N, B, D_dec), (N, B, D_dec) or None) or None
            idx: State ID to extract.

        Returns:
            : Decoder hidden state for given ID.
                ((N, 1, D_dec), (N, 1, D_dec) or None) or None

        """
        raise NotImplementedError
