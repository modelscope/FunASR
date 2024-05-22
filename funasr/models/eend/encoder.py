import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_units, h=8, dropout_rate=0.1):
        super().__init__()
        self.linearQ = nn.Linear(n_units, n_units)
        self.linearK = nn.Linear(n_units, n_units)
        self.linearV = nn.Linear(n_units, n_units)
        self.linearO = nn.Linear(n_units, n_units)
        self.d_k = n_units // h
        self.h = h
        self.dropout = nn.Dropout(dropout_rate)

    def __call__(self, x, batch_size, x_mask):
        q = self.linearQ(x).view(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).view(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).view(batch_size, -1, self.h, self.d_k)
        scores = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) / math.sqrt(self.d_k)
        if x_mask is not None:
            x_mask = x_mask.unsqueeze(1)
            scores = scores.masked_fill(x_mask == 0, -1e9)
        self.att = F.softmax(scores, dim=3)
        p_att = self.dropout(self.att)
        x = torch.matmul(p_att, v.permute(0, 2, 1, 3))
        x = x.permute(0, 2, 1, 3).contiguous().view(-1, self.h * self.d_k)
        return self.linearO(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, n_units, d_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(n_units, d_units)
        self.linear2 = nn.Linear(d_units, n_units)
        self.dropout = nn.Dropout(dropout_rate)

    def __call__(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EENDOLATransformerEncoder(nn.Module):
    def __init__(
        self,
        idim: int,
        n_layers: int,
        n_units: int,
        e_units: int = 2048,
        h: int = 4,
        dropout_rate: float = 0.1,
        use_pos_emb: bool = False,
    ):
        super(EENDOLATransformerEncoder, self).__init__()
        self.linear_in = nn.Linear(idim, n_units)
        self.lnorm_in = nn.LayerNorm(n_units)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(n_layers):
            setattr(self, "{}{:d}".format("lnorm1_", i), nn.LayerNorm(n_units))
            setattr(self, "{}{:d}".format("self_att_", i), MultiHeadSelfAttention(n_units, h))
            setattr(self, "{}{:d}".format("lnorm2_", i), nn.LayerNorm(n_units))
            setattr(
                self,
                "{}{:d}".format("ff_", i),
                PositionwiseFeedForward(n_units, e_units, dropout_rate),
            )
        self.lnorm_out = nn.LayerNorm(n_units)

    def __call__(self, x, x_mask=None):
        BT_size = x.shape[0] * x.shape[1]
        e = self.linear_in(x.reshape(BT_size, -1))
        for i in range(self.n_layers):
            e = getattr(self, "{}{:d}".format("lnorm1_", i))(e)
            s = getattr(self, "{}{:d}".format("self_att_", i))(e, x.shape[0], x_mask)
            e = e + self.dropout(s)
            e = getattr(self, "{}{:d}".format("lnorm2_", i))(e)
            s = getattr(self, "{}{:d}".format("ff_", i))(e)
            e = e + self.dropout(s)
        return self.lnorm_out(e)
