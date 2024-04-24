#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
from typing import List, Optional, Tuple

from funasr.register import tables
from funasr.models.specaug.specaug import SpecAug
from funasr.models.transducer.beam_search_transducer import Hypothesis


@tables.register("decoder_classes", "rnnt_decoder")
class RNNTDecoder(torch.nn.Module):
    """RNN decoder module.

    Args:
        vocab_size: Vocabulary size.
        embed_size: Embedding size.
        hidden_size: Hidden size..
        rnn_type: Decoder layers type.
        num_layers: Number of decoder layers.
        dropout_rate: Dropout rate for decoder layers.
        embed_dropout_rate: Dropout rate for embedding layer.
        embed_pad: Embedding padding symbol ID.

    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 256,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        dropout_rate: float = 0.0,
        embed_dropout_rate: float = 0.0,
        embed_pad: int = 0,
        use_embed_mask: bool = False,
    ) -> None:
        """Construct a RNNDecoder object."""
        super().__init__()

        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"Not supported: rnn_type={rnn_type}")

        self.embed = torch.nn.Embedding(vocab_size, embed_size, padding_idx=embed_pad)
        self.dropout_embed = torch.nn.Dropout(p=embed_dropout_rate)

        rnn_class = torch.nn.LSTM if rnn_type == "lstm" else torch.nn.GRU

        self.rnn = torch.nn.ModuleList([rnn_class(embed_size, hidden_size, 1, batch_first=True)])

        for _ in range(1, num_layers):
            self.rnn += [rnn_class(hidden_size, hidden_size, 1, batch_first=True)]

        self.dropout_rnn = torch.nn.ModuleList(
            [torch.nn.Dropout(p=dropout_rate) for _ in range(num_layers)]
        )

        self.dlayers = num_layers
        self.dtype = rnn_type

        self.output_size = hidden_size
        self.vocab_size = vocab_size

        self.device = next(self.parameters()).device
        self.score_cache = {}

        self.use_embed_mask = use_embed_mask
        if self.use_embed_mask:
            self._embed_mask = SpecAug(
                time_mask_width_range=3,
                num_time_mask=4,
                apply_freq_mask=False,
                apply_time_warp=False,
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
            states: Decoder hidden states.
                      ((N, B, D_dec), (N, B, D_dec) or None) or None

        Returns:
            dec_out: Decoder output sequences. (B, U, D_dec)

        """
        if states is None:
            states = self.init_state(labels.size(0))

        dec_embed = self.dropout_embed(self.embed(labels))
        if self.use_embed_mask and self.training:
            dec_embed = self._embed_mask(dec_embed, label_lens)[0]
        dec_out, states = self.rnn_forward(dec_embed, states)
        return dec_out

    def rnn_forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Encode source label sequences.

        Args:
            x: RNN input sequences. (B, D_emb)
            state: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec) or None)

        Returns:
            x: RNN output sequences. (B, D_dec)
            (h_next, c_next): Decoder hidden states.
                                (N, B, D_dec), (N, B, D_dec) or None)

        """
        h_prev, c_prev = state
        h_next, c_next = self.init_state(x.size(0))

        for layer in range(self.dlayers):
            if self.dtype == "lstm":
                x, (h_next[layer : layer + 1], c_next[layer : layer + 1]) = self.rnn[layer](
                    x, hx=(h_prev[layer : layer + 1], c_prev[layer : layer + 1])
                )
            else:
                x, h_next[layer : layer + 1] = self.rnn[layer](x, hx=h_prev[layer : layer + 1])

            x = self.dropout_rnn[layer](x)

        return x, (h_next, c_next)

    def score(
        self,
        label: torch.Tensor,
        label_sequence: List[int],
        dec_state: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """One-step forward hypothesis.

        Args:
            label: Previous label. (1, 1)
            label_sequence: Current label sequence.
            dec_state: Previous decoder hidden states.
                         ((N, 1, D_dec), (N, 1, D_dec) or None)

        Returns:
            dec_out: Decoder output sequence. (1, D_dec)
            dec_state: Decoder hidden states.
                         ((N, 1, D_dec), (N, 1, D_dec) or None)

        """
        str_labels = "_".join(map(str, label_sequence))

        if str_labels in self.score_cache:
            dec_out, dec_state = self.score_cache[str_labels]
        else:
            dec_embed = self.embed(label)
            dec_out, dec_state = self.rnn_forward(dec_embed, dec_state)

            self.score_cache[str_labels] = (dec_out, dec_state)

        return dec_out[0], dec_state

    def batch_score(
        self,
        hyps: List[Hypothesis],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.

        Returns:
            dec_out: Decoder output sequences. (B, D_dec)
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec) or None)

        """
        labels = torch.LongTensor([[h.yseq[-1]] for h in hyps], device=self.device)
        dec_embed = self.embed(labels)

        states = self.create_batch_states([h.dec_state for h in hyps])
        dec_out, states = self.rnn_forward(dec_embed, states)

        return dec_out.squeeze(1), states

    def set_device(self, device: torch.device) -> None:
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        self.device = device

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor, Optional[torch.tensor]]:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            : Initial decoder hidden states. ((N, B, D_dec), (N, B, D_dec) or None)

        """
        h_n = torch.zeros(
            self.dlayers,
            batch_size,
            self.output_size,
            device=self.device,
        )

        if self.dtype == "lstm":
            c_n = torch.zeros(
                self.dlayers,
                batch_size,
                self.output_size,
                device=self.device,
            )

            return (h_n, c_n)

        return (h_n, None)

    def select_state(
        self, states: Tuple[torch.Tensor, Optional[torch.Tensor]], idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get specified ID state from decoder hidden states.

        Args:
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec) or None)
            idx: State ID to extract.

        Returns:
            : Decoder hidden state for given ID. ((N, 1, D_dec), (N, 1, D_dec) or None)

        """
        return (
            states[0][:, idx : idx + 1, :],
            states[1][:, idx : idx + 1, :] if self.dtype == "lstm" else None,
        )

    def create_batch_states(
        self,
        new_states: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Create decoder hidden states.

        Args:
            new_states: Decoder hidden states. [N x ((1, D_dec), (1, D_dec) or None)]

        Returns:
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec) or None)

        """
        return (
            torch.cat([s[0] for s in new_states], dim=1),
            torch.cat([s[1] for s in new_states], dim=1) if self.dtype == "lstm" else None,
        )
