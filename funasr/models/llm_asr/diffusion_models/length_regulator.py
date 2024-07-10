import torch
from typing import Tuple
import torch.nn as nn
from torch.nn import functional as F
from funasr.models.llm_asr.diffusion_models.matcha_decoder import Upsample1D, Downsample1D
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from einops import repeat, pack
import logging


class UpSamplingRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling_ratios: Tuple,
            out_channels: int = None,
            groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for ratio in sampling_ratios:
                if ratio > 1:
                    module = Upsample1D(channels=channels, channel_first=False)
                else:
                    module = nn.Linear(channels, channels)
                norm = nn.LayerNorm(channels)
                act = nn.LeakyReLU()
                model.extend([module, norm, act])
        model.append(
            nn.Linear(channels, out_channels)
        )
        self.model = nn.Sequential(*model)

    def forward(self, x, xlens, y=None, y_lens=None, cond=None):
        # x, out, y in (B, T, D)
        out = self.model(x)
        out = out[:, :y.shape[1]]
        olens = y_lens

        return out, olens


class DownSamplingRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling_ratios: Tuple,
            out_channels: int = None,
            groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for ratio in sampling_ratios:
                if ratio > 1:
                    module = Downsample1D(dim=channels, channel_first=False, padding=2)
                else:
                    module = nn.Linear(channels, channels)
                norm = nn.LayerNorm(channels)
                act = nn.LeakyReLU()
                model.extend([module, norm, act])

        model.append(
            nn.Linear(channels, out_channels)
        )
        self.model = nn.Sequential(*model)

    def forward(self, x, xlens, y=None, y_lens=None, cond=None):
        # x, out, y in (B, T, D)
        out = self.model(x)
        out = out[:, :y.shape[1]]
        olens = y_lens

        return out, olens


class InterpolateRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling_ratios: Tuple,
            out_channels: int = None,
            groups: int = 1,
            mode="nearest",
            align_corners=False,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])

        model.append(
            nn.Conv1d(channels, out_channels, 1, 1)
        )
        self.model = nn.Sequential(*model)
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, xlens, y=None, ylens=None, cond=None):
        # x in (B, T, D)
        mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1)
        align_corners_opt = {}
        if self.mode in ["linear", "bilinear","bicubic", "trilinear"]:
            align_corners_opt = dict(align_corners=self.align_corners)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=y.shape[1],
                          mode=self.mode, **align_corners_opt)
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens

        return out * mask, olens


class UpsamplingBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            channels: int,
            stride=2,
            groups=1,
            channel_first=False,
    ):
        super().__init__()
        self.channel_first = channel_first
        self.stride = stride

        self.up_conv = nn.ConvTranspose1d(in_channels, channels, stride * 2, stride, 1)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv1d(channels, channels, 3, padding=1),
            torch.nn.GroupNorm(groups, channels),
            nn.Mish(),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv1d(channels, channels, 3, padding=1),
            torch.nn.GroupNorm(groups, channels),
            nn.Mish(),
        )
        self.res_conv = torch.nn.Conv1d(channels, channels, 1)

    def forward(self, x, ilens):
        if not self.channel_first:
            x = x.transpose(1, 2)

        olens = ilens * self.stride
        o_masks = (~make_pad_mask(olens))[:, None, :].to(x)
        res = out = self.up_conv(x) * o_masks

        out = self.block1(out) * o_masks + out
        out = self.block2(out) * o_masks + out
        out = out + self.res_conv(res) * o_masks

        if not self.channel_first:
            out = out.transpose(1, 2)

        return out, olens


class RepeatLengthRegulator(torch.nn.Module):
    """Repeat Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or
    phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.

        """
        super().__init__()
        self.pad_value = pad_value

    def forward(self, xs, ds, alpha=1.0):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).

        """
        if alpha != 1.0:
            assert alpha > 0
            ds = torch.round(ds.float() * alpha).long()

        if ds.sum() == 0:
            logging.warning(
                "predicted durations includes all 0 sequences. "
                "fill the first element with 1."
            )
            # NOTE(kan-bayashi): This case must not be happened in teacher forcing.
            #   It will be happened in inference with a bad duration predictor.
            #   So we do not need to care the padded sequence case here.
            ds[ds.sum(dim=1).eq(0)] = 1

        repeat = [torch.repeat_interleave(x, d, dim=0) for x, d in zip(xs, ds)]
        return pad_list(repeat, self.pad_value)
