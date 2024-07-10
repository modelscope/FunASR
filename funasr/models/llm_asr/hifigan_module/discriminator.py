"""hifigan based dicriminator implementation.

This code is modified from https://github.com/jik876/hifi-gan and https://github.com/kan-bayashi/ParallelWaveGAN.

"""

import typing as tp

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv2d, AvgPool1d, Conv1d
from torch.nn.utils import weight_norm, spectral_norm

from funasr.models.llm_asr.hifigan_module import get_padding


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3,
                 use_spectral_norm=False, lrelu_slope=0.1):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.lrelu_slope = lrelu_slope

        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(
                Conv2d(
                    1,
                    32, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(
                    32,
                    128, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(
                    128,
                    512, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(
                    512,
                    1024, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 periods: tp.List[int] = [2, 3, 5, 7, 11]):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(p) for p in periods
        ])

    def forward(self, x: torch.Tensor, return_intermediates: bool = True):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each
                layer output tensors.

        """
        outs = []
        for f in self.discriminators:
            # outs += [f(x)]
            if return_intermediates:
                outs.append(f(x))
            else:
                outs.append(f(x)[0])

        return outs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, lrelu_slope=0.1):
        super(DiscriminatorS, self).__init__()
        self.lrelu_slope = lrelu_slope
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, in_channels: int = 1, nb_scales: int = 3):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, x: torch.Tensor, return_intermediates: bool = True):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each
                layer output tensors.

        """
        outs = []
        for i, f in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i - 1](x)
            if return_intermediates:
                outs.append(f(x))
            else:
                outs.append(f(x)[0])

        return outs


class DiscriminatorR(nn.Module):
    def __init__(
        self,
        stft_params: tp.List[int],
        lrelu_slope: float = 0.1,
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        self.stft_params = stft_params
        self.lrelu_slope = lrelu_slope
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.stft_params
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        spec = torch.stft(x, n_fft, hop_length=hop_length, win_length=win_length,
                          center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

        spec = torch.view_as_real(spec)  # [B, F, TT, 2]
        mag = torch.norm(spec, p=2, dim =-1) #[B, F, TT]

        return mag

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x).unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        fft_sizes: tp.List[int] = [1024, 2048, 512],
        hop_sizes: tp.List[int] = [120, 240, 50],
        win_lengths: tp.List[int] = [600, 1200, 240],
        lrelu_slope: float = 0.1,
    ):
        super().__init__()

        self.discriminators = nn.ModuleList()

        for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
            self.discriminators.append(DiscriminatorR([fft, hop, win], lrelu_slope))

    def forward(self, x: torch.Tensor, return_intermediates: bool = True):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each
                layer output tensors.

        """
        outs = []
        for f in self.discriminators:
            if return_intermediates:
                outs.append(f(x))
            else:
                outs.append(f(x)[0])

        return outs


class MultipleDiscriminator(nn.Module):
    def __init__(
            self,
            input_size: int = 1,
            disc_conf_list: tp.List[tp.Dict[str, tp.Any]] = None,
    ):
        super().__init__()

        self.support_disc_choices = dict(
            mpd=MultiPeriodDiscriminator,
            msd=MultiScaleDiscriminator,
            mrd=MultiResolutionDiscriminator,
        )

        self.discriminators = nn.ModuleList()
        self.discriminator_type_lst = []
        for args in disc_conf_list:
            assert "name" in args, "disc_conf must have `name` attr to specific disc type."
            disc_type = args.pop("name")
            assert disc_type in self.support_disc_choices, \
                "Unsupported discriminator type, only support {}".format(
                    ",".join(self.support_disc_choices.keys())
                )

            disc_class = self.support_disc_choices[disc_type]
            one_disc = disc_class(in_channels=input_size, **args)
            self.discriminators.append(one_disc)
            # add back to the args for dump config.yaml
            args["name"] = disc_type
            self.discriminator_type_lst.append(disc_type)

    def get_discriminator_type_lst(self) -> tp.List[str]:
        return self.discriminator_type_lst

    def forward(self, x, return_intermediates=True):
        retval = []
        for disc in self.discriminators:
            out = disc(x, return_intermediates=return_intermediates)
            if isinstance(out, tuple):
                retval.append(out)
            elif isinstance(out, list):
                retval.extend(out)
            else:
                raise TypeError("The return value of discriminator must be tuple or list[tuple]")

        return retval