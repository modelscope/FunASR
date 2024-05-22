"""Sequential implementation of Recurrent Neural Network Language Model."""

from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from funasr.train.abs_model import AbsLM


class SequentialRNNLM(AbsLM):
    """Sequential RNNLM.

    See also:
        https://github.com/pytorch/examples/blob/4581968193699de14b56527296262dd76ab43557/word_language_model/model.py

    """

    def __init__(
        self,
        vocab_size: int,
        unit: int = 650,
        nhid: int = None,
        nlayers: int = 2,
        dropout_rate: float = 0.0,
        tie_weights: bool = False,
        rnn_type: str = "lstm",
        ignore_id: int = 0,
    ):
        super().__init__()

        ninp = unit
        if nhid is None:
            nhid = unit
        rnn_type = rnn_type.upper()

        self.drop = nn.Dropout(dropout_rate)
        self.encoder = nn.Embedding(vocab_size, ninp, padding_idx=ignore_id)
        if rnn_type in ["LSTM", "GRU"]:
            rnn_class = getattr(nn, rnn_type)
            self.rnn = rnn_class(ninp, nhid, nlayers, dropout=dropout_rate, batch_first=True)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                    options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(
                ninp,
                nhid,
                nlayers,
                nonlinearity=nonlinearity,
                dropout=dropout_rate,
                batch_first=True,
            )
        self.decoder = nn.Linear(nhid, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models"
        # (Press & Wolf 2016) https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers:
        # A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError("When using the tied flag, nhid must be equal to emsize")
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def zero_state(self):
        """Initialize LM state filled with zero values."""
        if isinstance(self.rnn, torch.nn.LSTM):
            h = torch.zeros((self.nlayers, self.nhid), dtype=torch.float)
            c = torch.zeros((self.nlayers, self.nhid), dtype=torch.float)
            state = h, c
        else:
            state = torch.zeros((self.nlayers, self.nhid), dtype=torch.float)

        return state

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.contiguous().view(output.size(0) * output.size(1), output.size(2))
        )
        return (
            decoded.view(output.size(0), output.size(1), decoded.size(1)),
            hidden,
        )

    def score(
        self,
        y: torch.Tensor,
        state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Score new token.

        Args:
            y: 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x: 2D encoder feature that generates ys.

        Returns:
            Tuple of
                torch.float32 scores for next token (n_vocab)
                and next state for ys

        """
        y, new_state = self(y[-1].view(1, 1), state)
        logp = y.log_softmax(dim=-1).view(-1)
        return logp, new_state

    def batch_score(
        self, ys: torch.Tensor, states: torch.Tensor, xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        if states[0] is None:
            states = None
        elif isinstance(self.rnn, torch.nn.LSTM):
            # states: Batch x 2 x (Nlayers, Dim) -> 2 x (Nlayers, Batch, Dim)
            h = torch.stack([h for h, c in states], dim=1)
            c = torch.stack([c for h, c in states], dim=1)
            states = h, c
        else:
            # states: Batch x (Nlayers, Dim) -> (Nlayers, Batch, Dim)
            states = torch.stack(states, dim=1)

        ys, states = self(ys[:, -1:], states)
        # ys: (Batch, 1, Nvocab) -> (Batch, NVocab)
        assert ys.size(1) == 1, ys.shape
        ys = ys.squeeze(1)
        logp = ys.log_softmax(dim=-1)

        # state: Change to batch first
        if isinstance(self.rnn, torch.nn.LSTM):
            # h, c: (Nlayers, Batch, Dim)
            h, c = states
            # states: Batch x 2 x (Nlayers, Dim)
            states = [(h[:, i], c[:, i]) for i in range(h.size(1))]
        else:
            # states: (Nlayers, Batch, Dim) -> Batch x (Nlayers, Dim)
            states = [states[:, i] for i in range(states.size(1))]

        return logp, states
