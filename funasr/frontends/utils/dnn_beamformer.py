"""DNN beamformer module."""

from typing import Tuple

import torch
from torch.nn import functional as F

from funasr.frontends.utils.beamformer import apply_beamforming_vector
from funasr.frontends.utils.beamformer import get_mvdr_vector
from funasr.frontends.utils.beamformer import (
    get_power_spectral_density_matrix,  # noqa: H301
)
from funasr.frontends.utils.mask_estimator import MaskEstimator
from torch_complex.tensor import ComplexTensor


class DNN_Beamformer(torch.nn.Module):
    """DNN mask based Beamformer

    Citation:
        Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;
        https://arxiv.org/abs/1703.04783

    """

    def __init__(
        self,
        bidim,
        btype="blstmp",
        blayers=3,
        bunits=300,
        bprojs=320,
        bnmask=2,
        dropout_rate=0.0,
        badim=320,
        ref_channel: int = -1,
        beamformer_type="mvdr",
    ):
        super().__init__()
        self.mask = MaskEstimator(btype, bidim, blayers, bunits, bprojs, dropout_rate, nmask=bnmask)
        self.ref = AttentionReference(bidim, badim)
        self.ref_channel = ref_channel

        self.nmask = bnmask

        if beamformer_type != "mvdr":
            raise ValueError("Not supporting beamformer_type={}".format(beamformer_type))
        self.beamformer_type = beamformer_type

    def forward(
        self, data: ComplexTensor, ilens: torch.LongTensor
    ) -> Tuple[ComplexTensor, torch.LongTensor, ComplexTensor]:
        """The forward function

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            data (ComplexTensor): (B, T, C, F)
            ilens (torch.Tensor): (B,)
        Returns:
            enhanced (ComplexTensor): (B, T, F)
            ilens (torch.Tensor): (B,)

        """

        def apply_beamforming(data, ilens, psd_speech, psd_noise):
            # u: (B, C)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_speech, ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)), device=data.device)
                u[..., self.ref_channel].fill_(1)

            ws = get_mvdr_vector(psd_speech, psd_noise, u)
            enhanced = apply_beamforming_vector(ws, data)

            return enhanced, ws

        # data (B, T, C, F) -> (B, F, C, T)
        data = data.permute(0, 3, 2, 1)

        # mask: (B, F, C, T)
        masks, _ = self.mask(data, ilens)
        assert self.nmask == len(masks)

        if self.nmask == 2:  # (mask_speech, mask_noise)
            mask_speech, mask_noise = masks

            psd_speech = get_power_spectral_density_matrix(data, mask_speech)
            psd_noise = get_power_spectral_density_matrix(data, mask_noise)

            enhanced, ws = apply_beamforming(data, ilens, psd_speech, psd_noise)

            # (..., F, T) -> (..., T, F)
            enhanced = enhanced.transpose(-1, -2)
            mask_speech = mask_speech.transpose(-1, -3)
        else:  # multi-speaker case: (mask_speech1, ..., mask_noise)
            mask_speech = list(masks[:-1])
            mask_noise = masks[-1]

            psd_speeches = [get_power_spectral_density_matrix(data, mask) for mask in mask_speech]
            psd_noise = get_power_spectral_density_matrix(data, mask_noise)

            enhanced = []
            ws = []
            for i in range(self.nmask - 1):
                psd_speech = psd_speeches.pop(i)
                # treat all other speakers' psd_speech as noises
                enh, w = apply_beamforming(data, ilens, psd_speech, sum(psd_speeches) + psd_noise)
                psd_speeches.insert(i, psd_speech)

                # (..., F, T) -> (..., T, F)
                enh = enh.transpose(-1, -2)
                mask_speech[i] = mask_speech[i].transpose(-1, -3)

                enhanced.append(enh)
                ws.append(w)

        return enhanced, ilens, mask_speech


class AttentionReference(torch.nn.Module):
    def __init__(self, bidim, att_dim):
        super().__init__()
        self.mlp_psd = torch.nn.Linear(bidim, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

    def forward(
        self, psd_in: ComplexTensor, ilens: torch.LongTensor, scaling: float = 2.0
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """The forward function

        Args:
            psd_in (ComplexTensor): (B, F, C, C)
            ilens (torch.Tensor): (B,)
            scaling (float):
        Returns:
            u (torch.Tensor): (B, C)
            ilens (torch.Tensor): (B,)
        """
        B, _, C = psd_in.size()[:3]
        assert psd_in.size(2) == psd_in.size(3), psd_in.size()
        # psd_in: (B, F, C, C)
        psd = psd_in.masked_fill(torch.eye(C, dtype=torch.bool, device=psd_in.device), 0)
        # psd: (B, F, C, C) -> (B, C, F)
        psd = (psd.sum(dim=-1) / (C - 1)).transpose(-1, -2)

        # Calculate amplitude
        psd_feat = (psd.real**2 + psd.imag**2) ** 0.5

        # (B, C, F) -> (B, C, F2)
        mlp_psd = self.mlp_psd(psd_feat)
        # (B, C, F2) -> (B, C, 1) -> (B, C)
        e = self.gvec(torch.tanh(mlp_psd)).squeeze(-1)
        u = F.softmax(scaling * e, dim=-1)
        return u, ilens
