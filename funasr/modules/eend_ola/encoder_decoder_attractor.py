import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class EncoderDecoderAttractor(nn.Module):

    def __init__(self, n_units, encoder_dropout=0.1, decoder_dropout=0.1):
        super(EncoderDecoderAttractor, self).__init__()
        self.enc0_dropout = nn.Dropout(encoder_dropout)
        self.encoder = nn.LSTM(n_units, n_units, 1, batch_first=True, dropout=encoder_dropout)
        self.dec0_dropout = nn.Dropout(decoder_dropout)
        self.decoder = nn.LSTM(n_units, n_units, 1, batch_first=True, dropout=decoder_dropout)
        self.counter = nn.Linear(n_units, 1)
        self.n_units = n_units

    def forward_core(self, xs, zeros):
        ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).to(torch.int64)
        xs = [self.enc0_dropout(x) for x in xs]
        xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=-1)
        xs = nn.utils.rnn.pack_padded_sequence(xs, ilens, batch_first=True, enforce_sorted=False)
        _, (hx, cx) = self.encoder(xs)
        zlens = torch.from_numpy(np.array([z.shape[0] for z in zeros])).to(torch.int64)
        max_zlen = torch.max(zlens).to(torch.int).item()
        zeros = [self.enc0_dropout(z) for z in zeros]
        zeros = nn.utils.rnn.pad_sequence(zeros, batch_first=True, padding_value=-1)
        zeros = nn.utils.rnn.pack_padded_sequence(zeros, zlens, batch_first=True, enforce_sorted=False)
        attractors, (_, _) = self.decoder(zeros, (hx, cx))
        attractors = nn.utils.rnn.pad_packed_sequence(attractors, batch_first=True, padding_value=-1,
                                                      total_length=max_zlen)[0]
        attractors = [att[:zlens[i].to(torch.int).item()] for i, att in enumerate(attractors)]
        return attractors

    def forward(self, xs, n_speakers):
        zeros = [torch.zeros(n_spk + 1, self.n_units).to(torch.float32).to(xs[0].device) for n_spk in n_speakers]
        attractors = self.forward_core(xs, zeros)
        labels = torch.cat([torch.from_numpy(np.array([[1] * n_spk + [0]], np.float32)) for n_spk in n_speakers], dim=1)
        labels = labels.to(xs[0].device)
        logit = torch.cat([self.counter(att).view(-1, n_spk + 1) for att, n_spk in zip(attractors, n_speakers)], dim=1)
        loss = F.binary_cross_entropy(torch.sigmoid(logit), labels)

        attractors = [att[slice(0, att.shape[0] - 1)] for att in attractors]
        return loss, attractors

    def estimate(self, xs, max_n_speakers=15):
        zeros = [torch.zeros(max_n_speakers, self.n_units).to(torch.float32).to(xs[0].device) for _ in xs]
        attractors = self.forward_core(xs, zeros)
        probs = [torch.sigmoid(torch.flatten(self.counter(att))) for att in attractors]
        return attractors, probs
