"""Attention modules for RNN."""

import math
import six

import torch
import torch.nn.functional as F

from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.utils.nets_utils import to_device


def _apply_attention_constraint(e, last_attended_idx, backward_window=1, forward_window=3):
    """Apply monotonic attention constraint.

    This function apply the monotonic attention constraint
    introduced in `Deep Voice 3: Scaling
    Text-to-Speech with Convolutional Sequence Learning`_.

    Args:
        e (Tensor): Attention energy before applying softmax (1, T).
        last_attended_idx (int): The index of the inputs of the last attended [0, T].
        backward_window (int, optional): Backward window size in attention constraint.
        forward_window (int, optional): Forward window size in attetion constraint.

    Returns:
        Tensor: Monotonic constrained attention energy (1, T).

    .. _`Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning`:
        https://arxiv.org/abs/1710.07654

    """
    if e.size(0) != 1:
        raise NotImplementedError("Batch attention constraining is not yet supported.")
    backward_idx = last_attended_idx - backward_window
    forward_idx = last_attended_idx + forward_window
    if backward_idx > 0:
        e[:, :backward_idx] = -float("inf")
    if forward_idx < e.size(1):
        e[:, forward_idx:] = -float("inf")
    return e


class NoAtt(torch.nn.Module):
    """No attention"""

    def __init__(self):
        super(NoAtt, self).__init__()
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        """NoAtt forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B, T_max, D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights
        :rtype: torch.Tensor
        """
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # if no bias, 0 0-pad goes 0
            mask = 1.0 - make_pad_mask(enc_hs_len).float()
            att_prev = mask / mask.new(enc_hs_len).unsqueeze(-1)
            att_prev = att_prev.to(self.enc_h)
            self.c = torch.sum(self.enc_h * att_prev.view(batch, self.h_length, 1), dim=1)

        return self.c, att_prev


class AttDot(torch.nn.Module):
    """Dot product attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param bool han_mode: flag to swith on mode of hierarchical attention
        and not store pre_compute_enc_h
    """

    def __init__(self, eprojs, dunits, att_dim, han_mode=False):
        super(AttDot, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        """AttDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weight (B x T_max)
        :rtype: torch.Tensor
        """

        batch = enc_hs_pad.size(0)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = torch.tanh(self.mlp_enc(self.enc_h))

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        e = torch.sum(
            self.pre_compute_enc_h * torch.tanh(self.mlp_dec(dec_z)).view(batch, 1, self.att_dim),
            dim=2,
        )  # utt x frame

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float("inf"))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        return c, w


class AttAdd(torch.nn.Module):
    """Additive attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param bool han_mode: flag to swith on mode of hierarchical attention
        and not store pre_compute_enc_h
    """

    def __init__(self, eprojs, dunits, att_dim, han_mode=False):
        super(AttAdd, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        """AttAdd forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights (B x T_max)
        :rtype: torch.Tensor
        """

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float("inf"))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, w


class AttLoc(torch.nn.Module):
    """location-aware attention module.

    Reference: Attention-Based Models for Speech Recognition
        (https://arxiv.org/pdf/1506.07503.pdf)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param bool han_mode: flag to swith on mode of hierarchical attention
        and not store pre_compute_enc_h
    """

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts, han_mode=False):
        super(AttLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1,
            aconv_chans,
            (1, 2 * aconv_filts + 1),
            padding=(0, aconv_filts),
            bias=False,
        )
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(
        self,
        enc_hs_pad,
        enc_hs_len,
        dec_z,
        att_prev,
        scaling=2.0,
        last_attended_idx=None,
        backward_window=1,
        forward_window=3,
    ):
        """Calculate AttLoc forward propagation.

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attention weight (B x T_max)
        :param float scaling: scaling parameter before applying softmax
        :param torch.Tensor forward_window:
            forward window size when constraining attention
        :param int last_attended_idx: index of the inputs of the last attended
        :param int backward_window: backward window size in attention constraint
        :param int forward_window: forward window size in attetion constraint
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights (B x T_max)
        :rtype: torch.Tensor
        """
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # if no bias, 0 0-pad goes 0
            att_prev = 1.0 - make_pad_mask(enc_hs_len).to(device=dec_z.device, dtype=dec_z.dtype)
            att_prev = att_prev / att_prev.new(enc_hs_len).unsqueeze(-1)

        # att_prev: utt x frame -> utt x 1 x 1 x frame
        # -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE: consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float("inf"))

        # apply monotonic attention constraint (mainly for TTS)
        if last_attended_idx is not None:
            e = _apply_attention_constraint(e, last_attended_idx, backward_window, forward_window)

        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, w


class AttCov(torch.nn.Module):
    """Coverage mechanism attention

    Reference: Get To The Point: Summarization with Pointer-Generator Network
       (https://arxiv.org/abs/1704.04368)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param bool han_mode: flag to swith on mode of hierarchical attention
        and not store pre_compute_enc_h
    """

    def __init__(self, eprojs, dunits, att_dim, han_mode=False):
        super(AttCov, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.wvec = torch.nn.Linear(1, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_list, scaling=2.0):
        """AttCov forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param list att_prev_list: list of previous attention weight
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weights
        :rtype: list
        """

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev_list is None:
            # if no bias, 0 0-pad goes 0
            att_prev_list = to_device(enc_hs_pad, (1.0 - make_pad_mask(enc_hs_len).float()))
            att_prev_list = [att_prev_list / att_prev_list.new(enc_hs_len).unsqueeze(-1)]

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = sum(att_prev_list)
        # cov_vec: B x T => B x T x 1 => B x T x att_dim
        cov_vec = self.wvec(cov_vec.unsqueeze(-1))

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(cov_vec + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float("inf"))
        w = F.softmax(scaling * e, dim=1)
        att_prev_list += [w]

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, att_prev_list


class AttLoc2D(torch.nn.Module):
    """2D location-aware attention

    This attention is an extended version of location aware attention.
    It take not only one frame before attention weights,
    but also earlier frames into account.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param int att_win: attention window size (default=5)
    :param bool han_mode:
        flag to swith on mode of hierarchical attention and not store pre_compute_enc_h
    """

    def __init__(self, eprojs, dunits, att_dim, att_win, aconv_chans, aconv_filts, han_mode=False):
        super(AttLoc2D, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1,
            aconv_chans,
            (att_win, 2 * aconv_filts + 1),
            padding=(0, aconv_filts),
            bias=False,
        )
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans
        self.att_win = att_win
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        """AttLoc2D forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attention weight (B x att_win x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights (B x att_win x T_max)
        :rtype: torch.Tensor
        """

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # B * [Li x att_win]
            # if no bias, 0 0-pad goes 0
            att_prev = to_device(enc_hs_pad, (1.0 - make_pad_mask(enc_hs_len).float()))
            att_prev = att_prev / att_prev.new(enc_hs_len).unsqueeze(-1)
            att_prev = att_prev.unsqueeze(1).expand(-1, self.att_win, -1)

        # att_prev: B x att_win x Tmax -> B x 1 x att_win x Tmax -> B x C x 1 x Tmax
        att_conv = self.loc_conv(att_prev.unsqueeze(1))
        # att_conv: B x C x 1 x Tmax -> B x Tmax x C
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float("inf"))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        # update att_prev: B x att_win x Tmax -> B x att_win+1 x Tmax
        # -> B x att_win x Tmax
        att_prev = torch.cat([att_prev, w.unsqueeze(1)], dim=1)
        att_prev = att_prev[:, 1:]

        return c, att_prev


class AttLocRec(torch.nn.Module):
    """location-aware recurrent attention

    This attention is an extended version of location aware attention.
    With the use of RNN,
    it take the effect of the history of attention weights into account.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param bool han_mode:
        flag to swith on mode of hierarchical attention and not store pre_compute_enc_h
    """

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts, han_mode=False):
        super(AttLocRec, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1,
            aconv_chans,
            (1, 2 * aconv_filts + 1),
            padding=(0, aconv_filts),
            bias=False,
        )
        self.att_lstm = torch.nn.LSTMCell(aconv_chans, att_dim, bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_states, scaling=2.0):
        """AttLocRec forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param tuple att_prev_states: previous attention weight and lstm states
                                      ((B, T_max), ((B, att_dim), (B, att_dim)))
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights and lstm states (w, (hx, cx))
                 ((B, T_max), ((B, att_dim), (B, att_dim)))
        :rtype: tuple
        """

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev_states is None:
            # initialize attention weight with uniform dist.
            # if no bias, 0 0-pad goes 0
            att_prev = to_device(enc_hs_pad, (1.0 - make_pad_mask(enc_hs_len).float()))
            att_prev = att_prev / att_prev.new(enc_hs_len).unsqueeze(-1)

            # initialize lstm states
            att_h = enc_hs_pad.new_zeros(batch, self.att_dim)
            att_c = enc_hs_pad.new_zeros(batch, self.att_dim)
            att_states = (att_h, att_c)
        else:
            att_prev = att_prev_states[0]
            att_states = att_prev_states[1]

        # B x 1 x 1 x T -> B x C x 1 x T
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # apply non-linear
        att_conv = F.relu(att_conv)
        # B x C x 1 x T -> B x C x 1 x 1 -> B x C
        att_conv = F.max_pool2d(att_conv, (1, att_conv.size(3))).view(batch, -1)

        att_h, att_c = self.att_lstm(att_conv, att_states)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(
            torch.tanh(att_h.unsqueeze(1) + self.pre_compute_enc_h + dec_z_tiled)
        ).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float("inf"))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, (w, (att_h, att_c))


class AttCovLoc(torch.nn.Module):
    """Coverage mechanism location aware attention

    This attention is a combination of coverage and location-aware attentions.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param bool han_mode:
        flag to swith on mode of hierarchical attention and not store pre_compute_enc_h
    """

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts, han_mode=False):
        super(AttCovLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1,
            aconv_chans,
            (1, 2 * aconv_filts + 1),
            padding=(0, aconv_filts),
            bias=False,
        )
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_list, scaling=2.0):
        """AttCovLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param list att_prev_list: list of previous attention weight
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weights
        :rtype: list
        """

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev_list is None:
            # if no bias, 0 0-pad goes 0
            mask = 1.0 - make_pad_mask(enc_hs_len).float()
            att_prev_list = [to_device(enc_hs_pad, mask / mask.new(enc_hs_len).unsqueeze(-1))]

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = sum(att_prev_list)

        # cov_vec: B x T -> B x 1 x 1 x T -> B x C x 1 x T
        att_conv = self.loc_conv(cov_vec.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float("inf"))
        w = F.softmax(scaling * e, dim=1)
        att_prev_list += [w]

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, att_prev_list


class AttMultiHeadDot(torch.nn.Module):
    """Multi head dot product attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param bool han_mode: flag to swith on mode of hierarchical attention
        and not store pre_compute_k and pre_compute_v
    """

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v, han_mode=False):
        super(AttMultiHeadDot, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        for _ in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        """AttMultiHeadDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :return: attention weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weight (B x T_max) * aheads
        :rtype: list
        """

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                torch.tanh(self.mlp_k[h](self.enc_h)) for h in six.moves.range(self.aheads)
            ]

        if self.pre_compute_v is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            e = torch.sum(
                self.pre_compute_k[h]
                * torch.tanh(self.mlp_q[h](dec_z)).view(batch, 1, self.att_dim_k),
                dim=2,
            )  # utt x frame

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float("inf"))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadAdd(torch.nn.Module):
    """Multi head additive attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using additive attention for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param bool han_mode: flag to swith on mode of hierarchical attention
        and not store pre_compute_k and pre_compute_v
    """

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v, han_mode=False):
        super(AttMultiHeadAdd, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        for _ in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        """AttMultiHeadAdd forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weight (B x T_max) * aheads
        :rtype: list
        """

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            e = self.gvec[h](
                torch.tanh(
                    self.pre_compute_k[h] + self.mlp_q[h](dec_z).view(batch, 1, self.att_dim_k)
                )
            ).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float("inf"))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadLoc(torch.nn.Module):
    """Multi head location based attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using location-aware attention for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param bool han_mode: flag to swith on mode of hierarchical attention
        and not store pre_compute_k and pre_compute_v
    """

    def __init__(
        self,
        eprojs,
        dunits,
        aheads,
        att_dim_k,
        att_dim_v,
        aconv_chans,
        aconv_filts,
        han_mode=False,
    ):
        super(AttMultiHeadLoc, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        self.loc_conv = torch.nn.ModuleList()
        self.mlp_att = torch.nn.ModuleList()
        for _ in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
            self.loc_conv += [
                torch.nn.Conv2d(
                    1,
                    aconv_chans,
                    (1, 2 * aconv_filts + 1),
                    padding=(0, aconv_filts),
                    bias=False,
                )
            ]
            self.mlp_att += [torch.nn.Linear(aconv_chans, att_dim_k, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        """AttMultiHeadLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev:
            list of previous attention weight (B x T_max) * aheads
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weight (B x T_max) * aheads
        :rtype: list
        """

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for _ in six.moves.range(self.aheads):
                # if no bias, 0 0-pad goes 0
                mask = 1.0 - make_pad_mask(enc_hs_len).float()
                att_prev += [to_device(enc_hs_pad, mask / mask.new(enc_hs_len).unsqueeze(-1))]

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = self.mlp_att[h](att_conv)

            e = self.gvec[h](
                torch.tanh(
                    self.pre_compute_k[h]
                    + att_conv
                    + self.mlp_q[h](dec_z).view(batch, 1, self.att_dim_k)
                )
            ).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float("inf"))
            w += [F.softmax(scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadMultiResLoc(torch.nn.Module):
    """Multi head multi resolution location based attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using location-aware attention for each head.
    Furthermore, it uses different filter size for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param int aconv_chans: maximum # channels of attention convolution
        each head use #ch = aconv_chans * (head + 1) / aheads
        e.g. aheads=4, aconv_chans=100 => filter size = 25, 50, 75, 100
    :param int aconv_filts: filter size of attention convolution
    :param bool han_mode: flag to swith on mode of hierarchical attention
        and not store pre_compute_k and pre_compute_v
    """

    def __init__(
        self,
        eprojs,
        dunits,
        aheads,
        att_dim_k,
        att_dim_v,
        aconv_chans,
        aconv_filts,
        han_mode=False,
    ):
        super(AttMultiHeadMultiResLoc, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        self.loc_conv = torch.nn.ModuleList()
        self.mlp_att = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
            afilts = aconv_filts * (h + 1) // aheads
            self.loc_conv += [
                torch.nn.Conv2d(
                    1, aconv_chans, (1, 2 * afilts + 1), padding=(0, afilts), bias=False
                )
            ]
            self.mlp_att += [torch.nn.Linear(aconv_chans, att_dim_k, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        """AttMultiHeadMultiResLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: list of previous attention weight
            (B x T_max) * aheads
        :return: attention weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attention weight (B x T_max) * aheads
        :rtype: list
        """

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for _ in six.moves.range(self.aheads):
                # if no bias, 0 0-pad goes 0
                mask = 1.0 - make_pad_mask(enc_hs_len).float()
                att_prev += [to_device(enc_hs_pad, mask / mask.new(enc_hs_len).unsqueeze(-1))]

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = self.mlp_att[h](att_conv)

            e = self.gvec[h](
                torch.tanh(
                    self.pre_compute_k[h]
                    + att_conv
                    + self.mlp_q[h](dec_z).view(batch, 1, self.att_dim_k)
                )
            ).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float("inf"))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttForward(torch.nn.Module):
    """Forward attention module.

    Reference:
    Forward attention in sequence-to-sequence acoustic modeling for speech synthesis
        (https://arxiv.org/pdf/1807.06736.pdf)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    """

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttForward, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1,
            aconv_chans,
            (1, 2 * aconv_filts + 1),
            padding=(0, aconv_filts),
            bias=False,
        )
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(
        self,
        enc_hs_pad,
        enc_hs_len,
        dec_z,
        att_prev,
        scaling=1.0,
        last_attended_idx=None,
        backward_window=1,
        forward_window=3,
    ):
        """Calculate AttForward forward propagation.

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: attention weights of previous step
        :param float scaling: scaling parameter before applying softmax
        :param int last_attended_idx: index of the inputs of the last attended
        :param int backward_window: backward window size in attention constraint
        :param int forward_window: forward window size in attetion constraint
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights (B x T_max)
        :rtype: torch.Tensor
        """
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            # initial attention will be [1, 0, 0, ...]
            att_prev = enc_hs_pad.new_zeros(*enc_hs_pad.size()[:2])
            att_prev[:, 0] = 1.0

        # att_prev: utt x frame -> utt x 1 x 1 x frame
        # -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).unsqueeze(1)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled + att_conv)).squeeze(2)

        # NOTE: consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float("inf"))

        # apply monotonic attention constraint (mainly for TTS)
        if last_attended_idx is not None:
            e = _apply_attention_constraint(e, last_attended_idx, backward_window, forward_window)

        w = F.softmax(scaling * e, dim=1)

        # forward attention
        att_prev_shift = F.pad(att_prev, (1, 0))[:, :-1]
        w = (att_prev + att_prev_shift) * w
        # NOTE: clamp is needed to avoid nan gradient
        w = F.normalize(torch.clamp(w, 1e-6), p=1, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.unsqueeze(-1), dim=1)

        return c, w


class AttForwardTA(torch.nn.Module):
    """Forward attention with transition agent module.

    Reference:
    Forward attention in sequence-to-sequence acoustic modeling for speech synthesis
        (https://arxiv.org/pdf/1807.06736.pdf)

    :param int eunits: # units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param int odim: output dimension
    """

    def __init__(self, eunits, dunits, att_dim, aconv_chans, aconv_filts, odim):
        super(AttForwardTA, self).__init__()
        self.mlp_enc = torch.nn.Linear(eunits, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_ta = torch.nn.Linear(eunits + dunits + odim, 1)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1,
            aconv_chans,
            (1, 2 * aconv_filts + 1),
            padding=(0, aconv_filts),
            bias=False,
        )
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eunits = eunits
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.trans_agent_prob = 0.5

    def reset(self):
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.trans_agent_prob = 0.5

    def forward(
        self,
        enc_hs_pad,
        enc_hs_len,
        dec_z,
        att_prev,
        out_prev,
        scaling=1.0,
        last_attended_idx=None,
        backward_window=1,
        forward_window=3,
    ):
        """Calculate AttForwardTA forward propagation.

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B, Tmax, eunits)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: decoder hidden state (B, dunits)
        :param torch.Tensor att_prev: attention weights of previous step
        :param torch.Tensor out_prev: decoder outputs of previous step (B, odim)
        :param float scaling: scaling parameter before applying softmax
        :param int last_attended_idx: index of the inputs of the last attended
        :param int backward_window: backward window size in attention constraint
        :param int forward_window: forward window size in attetion constraint
        :return: attention weighted encoder state (B, dunits)
        :rtype: torch.Tensor
        :return: previous attention weights (B, Tmax)
        :rtype: torch.Tensor
        """
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            # initial attention will be [1, 0, 0, ...]
            att_prev = enc_hs_pad.new_zeros(*enc_hs_pad.size()[:2])
            att_prev[:, 0] = 1.0

        # att_prev: utt x frame -> utt x 1 x 1 x frame
        # -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(enc_hs_pad, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float("inf"))

        # apply monotonic attention constraint (mainly for TTS)
        if last_attended_idx is not None:
            e = _apply_attention_constraint(e, last_attended_idx, backward_window, forward_window)

        w = F.softmax(scaling * e, dim=1)

        # forward attention
        att_prev_shift = F.pad(att_prev, (1, 0))[:, :-1]
        w = (self.trans_agent_prob * att_prev + (1 - self.trans_agent_prob) * att_prev_shift) * w
        # NOTE: clamp is needed to avoid nan gradient
        w = F.normalize(torch.clamp(w, 1e-6), p=1, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        # update transition agent prob
        self.trans_agent_prob = torch.sigmoid(self.mlp_ta(torch.cat([c, out_prev, dec_z], dim=1)))

        return c, w


def att_for(args, num_att=1, han_mode=False):
    """Instantiates an attention module given the program arguments

    :param Namespace args: The arguments
    :param int num_att: number of attention modules
        (in multi-speaker case, it can be 2 or more)
    :param bool han_mode: switch on/off mode of hierarchical attention network (HAN)
    :rtype torch.nn.Module
    :return: The attention module
    """
    att_list = torch.nn.ModuleList()
    num_encs = getattr(args, "num_encs", 1)  # use getattr to keep compatibility
    aheads = getattr(args, "aheads", None)
    awin = getattr(args, "awin", None)
    aconv_chans = getattr(args, "aconv_chans", None)
    aconv_filts = getattr(args, "aconv_filts", None)

    if num_encs == 1:
        for i in range(num_att):
            att = initial_att(
                args.atype,
                args.eprojs,
                args.dunits,
                aheads,
                args.adim,
                awin,
                aconv_chans,
                aconv_filts,
            )
            att_list.append(att)
    elif num_encs > 1:  # no multi-speaker mode
        if han_mode:
            att = initial_att(
                args.han_type,
                args.eprojs,
                args.dunits,
                args.han_heads,
                args.han_dim,
                args.han_win,
                args.han_conv_chans,
                args.han_conv_filts,
                han_mode=True,
            )
            return att
        else:
            att_list = torch.nn.ModuleList()
            for idx in range(num_encs):
                att = initial_att(
                    args.atype[idx],
                    args.eprojs,
                    args.dunits,
                    aheads[idx],
                    args.adim[idx],
                    awin[idx],
                    aconv_chans[idx],
                    aconv_filts[idx],
                )
                att_list.append(att)
    else:
        raise ValueError("Number of encoders needs to be more than one. {}".format(num_encs))
    return att_list


def initial_att(
    atype, eprojs, dunits, aheads, adim, awin, aconv_chans, aconv_filts, han_mode=False
):
    """Instantiates a single attention module

    :param str atype: attention type
    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int adim: attention dimension
    :param int awin: attention window size
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param bool han_mode: flag to swith on mode of hierarchical attention
    :return: The attention module
    """

    if atype == "noatt":
        att = NoAtt()
    elif atype == "dot":
        att = AttDot(eprojs, dunits, adim, han_mode)
    elif atype == "add":
        att = AttAdd(eprojs, dunits, adim, han_mode)
    elif atype == "location":
        att = AttLoc(eprojs, dunits, adim, aconv_chans, aconv_filts, han_mode)
    elif atype == "location2d":
        att = AttLoc2D(eprojs, dunits, adim, awin, aconv_chans, aconv_filts, han_mode)
    elif atype == "location_recurrent":
        att = AttLocRec(eprojs, dunits, adim, aconv_chans, aconv_filts, han_mode)
    elif atype == "coverage":
        att = AttCov(eprojs, dunits, adim, han_mode)
    elif atype == "coverage_location":
        att = AttCovLoc(eprojs, dunits, adim, aconv_chans, aconv_filts, han_mode)
    elif atype == "multi_head_dot":
        att = AttMultiHeadDot(eprojs, dunits, aheads, adim, adim, han_mode)
    elif atype == "multi_head_add":
        att = AttMultiHeadAdd(eprojs, dunits, aheads, adim, adim, han_mode)
    elif atype == "multi_head_loc":
        att = AttMultiHeadLoc(
            eprojs, dunits, aheads, adim, adim, aconv_chans, aconv_filts, han_mode
        )
    elif atype == "multi_head_multi_res_loc":
        att = AttMultiHeadMultiResLoc(
            eprojs, dunits, aheads, adim, adim, aconv_chans, aconv_filts, han_mode
        )
    return att


def att_to_numpy(att_ws, att):
    """Converts attention weights to a numpy array given the attention

    :param list att_ws: The attention weights
    :param torch.nn.Module att: The attention
    :rtype: np.ndarray
    :return: The numpy array of the attention weights
    """
    # convert to numpy array with the shape (B, Lmax, Tmax)
    if isinstance(att, AttLoc2D):
        # att_ws => list of previous concate attentions
        att_ws = torch.stack([aw[:, -1] for aw in att_ws], dim=1).cpu().numpy()
    elif isinstance(att, (AttCov, AttCovLoc)):
        # att_ws => list of list of previous attentions
        att_ws = torch.stack([aw[idx] for idx, aw in enumerate(att_ws)], dim=1).cpu().numpy()
    elif isinstance(att, AttLocRec):
        # att_ws => list of tuple of attention and hidden states
        att_ws = torch.stack([aw[0] for aw in att_ws], dim=1).cpu().numpy()
    elif isinstance(
        att,
        (AttMultiHeadDot, AttMultiHeadAdd, AttMultiHeadLoc, AttMultiHeadMultiResLoc),
    ):
        # att_ws => list of list of each head attention
        n_heads = len(att_ws[0])
        att_ws_sorted_by_head = []
        for h in six.moves.range(n_heads):
            att_ws_head = torch.stack([aw[h] for aw in att_ws], dim=1)
            att_ws_sorted_by_head += [att_ws_head]
        att_ws = torch.stack(att_ws_sorted_by_head, dim=1).cpu().numpy()
    else:
        # att_ws => list of attentions
        att_ws = torch.stack(att_ws, dim=1).cpu().numpy()
    return att_ws
