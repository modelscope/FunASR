"""Stateless decoder definition for Transducer models."""

from typing import List, Optional, Tuple

import torch
from typeguard import check_argument_types

from funasr.models_transducer.beam_search_transducer import Hypothesis
from funasr.models_transducer.decoder.abs_decoder import AbsDecoder
from funasr.models.specaug.specaug import SpecAug

class StatelessDecoder(AbsDecoder):
    """Stateless Transducer decoder module.

    Args:
        vocab_size: Output size.
        embed_size: Embedding size.
        embed_dropout_rate: Dropout rate for embedding layer.
        embed_pad: Embed/Blank symbol ID.

    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        embed_dropout_rate: float = 0.0,
        embed_pad: int = 0,
        use_embed_mask: bool = False,
    ) -> None:
        """Construct a StatelessDecoder object."""
        super().__init__()

        assert check_argument_types()

        self.embed = torch.nn.Embedding(vocab_size, embed_size, padding_idx=embed_pad)
        self.embed_dropout_rate = torch.nn.Dropout(p=embed_dropout_rate)

        self.output_size = embed_size
        self.vocab_size = vocab_size

        self.device = next(self.parameters()).device
        self.score_cache = {}

        self.use_embed_mask = use_embed_mask
        if self.use_embed_mask:
            self._embed_mask = SpecAug(
                time_mask_width_range=3,
                num_time_mask=1,
                apply_freq_mask=False,
                apply_time_warp=False
            )


    def forward(
        self,
        labels: torch.Tensor,
        label_lens: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Encode source label sequences.

        Args:
            labels: Label ID sequences. (B, L)
            states: Decoder hidden states. None

        Returns:
            dec_embed: Decoder output sequences. (B, U, D_emb)

        """
        dec_embed = self.embed_dropout_rate(self.embed(labels))
        if self.use_embed_mask and self.training:
            dec_embed = self._embed_mask(dec_embed, label_lens)[0]

        return dec_embed

    def score(
        self,
        label: torch.Tensor,
        label_sequence: List[int],
        state: None,
    ) -> Tuple[torch.Tensor, None]:
        """One-step forward hypothesis.

        Args:
            label: Previous label. (1, 1)
            label_sequence: Current label sequence.
            state: Previous decoder hidden states. None

        Returns:
            dec_out: Decoder output sequence. (1, D_emb)
            state: Decoder hidden states. None

        """
        str_labels = "_".join(map(str, label_sequence))

        if str_labels in self.score_cache:
            dec_embed = self.score_cache[str_labels]
        else:
            dec_embed = self.embed(label)

            self.score_cache[str_labels] = dec_embed

        return dec_embed[0], None

    def batch_score(
        self,
        hyps: List[Hypothesis],
    ) -> Tuple[torch.Tensor, None]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.

        Returns:
            dec_out: Decoder output sequences. (B, D_dec)
            states: Decoder hidden states. None

        """
        labels = torch.LongTensor([[h.yseq[-1]] for h in hyps], device=self.device)
        dec_embed = self.embed(labels)

        return dec_embed.squeeze(1), None

    def set_device(self, device: torch.device) -> None:
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        self.device = device

    def init_state(self, batch_size: int) -> None:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            : Initial decoder hidden states. None

        """
        return None

    def select_state(self, states: Optional[torch.Tensor], idx: int) -> None:
        """Get specified ID state from decoder hidden states.

        Args:
            states: Decoder hidden states. None
            idx: State ID to extract.

        Returns:
            : Decoder hidden state for given ID. None

        """
        return None
