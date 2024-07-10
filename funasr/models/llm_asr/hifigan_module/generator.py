"""hifigan based generator implementation.

This code is modified from https://github.com/jik876/hifi-gan
 ,https://github.com/kan-bayashi/ParallelWaveGAN and
 https://github.com/NVIDIA/BigVGAN

"""

import typing as tp

import numpy as np
from scipy.signal import get_window
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm
from torch.nn.utils import remove_weight_norm

from funasr.models.llm_asr.hifigan_module import get_padding, init_weights
from funasr.models.llm_asr.hifigan_module.activations import Snake, SnakeBeta
from funasr.models.llm_asr.hifigan_module.nsf_utils import SourceModule, SourceModuleHnNSF


class ResBlock(torch.nn.Module):
    """Residual block module in HiFiGAN/BigVGAN."""
    def __init__(
        self,
        channels: int = 512,
        kernel_size: int = 3,
        dilations: tp.List[int] = [1, 3, 5],
        use_additional_convs: bool = True,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: tp.Dict[str, tp.Any] = {"negative_slope": 0.1},
    ):
        super(ResBlock, self).__init__()
        self.use_additional_convs = use_additional_convs

        self.convs1 = nn.ModuleList()
        if use_additional_convs:
            self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation)
                    )
                )
            )

            if use_additional_convs:
                self.convs2.append(
                    weight_norm(
                        Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            padding=get_padding(kernel_size, 1)
                        )
                    )
                )

        self.convs1.apply(init_weights)
        if use_additional_convs:
            self.convs2.apply(init_weights)

        if nonlinear_activation == "LeakyReLU":
            self.activations1 = nn.ModuleList([
                nn.LeakyReLU(nonlinear_activation_params["negative_slope"])
                for _ in range(len(self.convs1))
            ])
            if use_additional_convs:
                self.activations2 = nn.ModuleList([
                    nn.LeakyReLU(nonlinear_activation_params["negative_slope"])
                    for _ in range(len(self.convs2))
                ])

        elif nonlinear_activation == "Snake":
            self.activations1 = nn.ModuleList([
                Snake(channels, alpha_logscale=nonlinear_activation_params.get("alpha_logscale", False))
                for _ in range(len(self.convs1))
            ])
            if use_additional_convs:
                self.activations2 = nn.ModuleList([
                    Snake(channels, alpha_logscale=nonlinear_activation_params.get("alpha_logscale", False))
                    for _ in range(len(self.convs2))
                ])

        elif nonlinear_activation == "SnakeBeta":
            self.activations1 = nn.ModuleList([
                SnakeBeta(channels, alpha_logscale=nonlinear_activation_params.get("alpha_logscale", False))
                for _ in range(len(self.convs1))
            ])
            if use_additional_convs:
                self.activations2 = nn.ModuleList([
                    SnakeBeta(channels, alpha_logscale=nonlinear_activation_params.get("alpha_logscale", False))
                    for _ in range(len(self.convs2))
                ])

        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            if self.use_additional_convs:
                xt = self.activations2[idx](xt)
                xt = self.convs2[idx](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for idx in range(len(self.convs1)):
            remove_weight_norm(self.convs1[idx])
            if self.use_additional_convs:
                remove_weight_norm(self.convs2[idx])


class HifiGenerator(nn.Module):
    def __init__(
            self,
            in_channels: int = 80,
            base_channels: int = 512,
            global_channels: int = -1,
            upsample_rates: tp.List[int] = [8, 8, 2, 2],
            upsample_kernel_sizes: tp.List[int] = [16, 16, 4, 4],
            resblock_kernel_sizes: tp.List[int] = [3, 7, 11],
            resblock_dilation_sizes: tp.List[tp.List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            resblock_nonlinear_activation: str = "LeakyReLU",
            resblock_nonlinear_activation_params: tp.Dict[str, tp.Any] = {"negative_slope": 0.1},
            use_additional_convs: bool = True,
            cond_in_each_up_layer: bool = False,
            lrelu_slope: float = 0.1,
            act_pre_each_up_layer: bool = True
    ):
        super(HifiGenerator, self).__init__()

        self.out_channels = 1
        self.global_channels = global_channels
        self.use_additional_convs = use_additional_convs
        self.cond_in_each_up_layer = cond_in_each_up_layer if global_channels > 0 else False
        self.lrelu_slope = lrelu_slope
        self.act_pre_each_up_layer = act_pre_each_up_layer

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(
            Conv1d(in_channels, base_channels, 7, 1, padding=3)
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        base_channels // (2**i),
                        base_channels // (2**(i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2**(i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d, use_additional_convs,
                                               resblock_nonlinear_activation,
                                               resblock_nonlinear_activation_params))

        if self.global_channels > 0:
            self.conv_global_cond = weight_norm(
                Conv1d(global_channels, base_channels, 1)
            )
            self.conv_global_cond.apply(init_weights)

            if self.cond_in_each_up_layer:
                self.conv_conds = nn.ModuleList()
                for i in range(len(self.ups)):
                    self.conv_conds.append(weight_norm(
                        nn.Conv1d(global_channels, base_channels // (2**(i + 1)), 1))
                    )
                self.conv_conds.apply(init_weights)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def output_size(self):
        return self.out_channels

    def forward(self, x: torch.Tensor, g: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        # x in (B, in_channels, T), g in (B, global_channels, 1)
        x = self.conv_pre(x)
        if self.global_channels > 0 and g is not None:
            x = x + self.conv_global_cond(g)

        for i in range(self.num_upsamples):
            if self.act_pre_each_up_layer:
                x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if self.cond_in_each_up_layer and g is not None:
                x = x + self.conv_conds[i](g)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        if self.global_channels > 0:
            remove_weight_norm(self.conv_global_cond)
        if self.cond_in_each_up_layer:
            for l in self.conv_conds:
                remove_weight_norm(l)


class NsfHifiGenerator(nn.Module):
    """
    Neural Source Filter + HifiGan
    """
    def __init__(
            self,
            in_channels: int = 80,
            base_channels: int = 512,
            global_channels: int = -1,
            nb_harmonics: int = 7,
            sampling_rate: int = 22050,
            nsf_alpha: float = 0.1,
            nsf_sigma: float = 0.003,
            nsf_voiced_threshold: float = 10,
            upsample_rates: tp.List[int] = [8, 8, 2, 2],
            upsample_kernel_sizes: tp.List[int] = [16, 16, 4, 4],
            resblock_kernel_sizes: tp.List[int] = [3, 7, 11],
            resblock_dilation_sizes: tp.List[tp.List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            resblock_nonlinear_activation: str = "LeakyReLU",
            resblock_nonlinear_activation_params: tp.Dict[str, tp.Any] = {"negative_slope": 0.1},
            use_additional_convs: bool = True,
            cond_in_each_up_layer: bool = False,
            lrelu_slope: float = 0.1,
            act_pre_each_up_layer: bool = True
    ):
        super(NsfHifiGenerator, self).__init__()

        self.out_channels = 1
        self.global_channels = global_channels
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.use_additional_convs = use_additional_convs
        self.cond_in_each_up_layer = cond_in_each_up_layer if global_channels > 0 else False
        self.lrelu_slope = lrelu_slope
        self.act_pre_each_up_layer = act_pre_each_up_layer

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.source_module = SourceModule(nb_harmonics, np.cumprod(upsample_rates)[-1],
                                          sampling_rate, nsf_alpha, nsf_sigma, nsf_voiced_threshold)
        self.conv_pre = weight_norm(
            Conv1d(in_channels, base_channels, 7, 1, padding=3)
        )

        # Up
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        base_channels // (2**i),
                        base_channels // (2**(i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        # Down
        self.source_downs = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, u in enumerate(downsample_cum_rates[::-1]):
            if (u == 1):
                self.source_downs.append(
                weight_norm(Conv1d(1, base_channels // (2 ** (i + 1)), 1, 1))
                )
            else:
                self.source_downs.append(
                weight_norm(Conv1d(1, base_channels // (2 ** (i + 1)), u*2, u, padding=(u//2)))
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2**(i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d, use_additional_convs,
                                               resblock_nonlinear_activation,
                                               resblock_nonlinear_activation_params))

        if self.global_channels > 0:
            self.conv_global_cond = weight_norm(
                Conv1d(global_channels, base_channels, 1)
            )
            self.conv_global_cond.apply(init_weights)

            if self.cond_in_each_up_layer:
                self.conv_conds = nn.ModuleList()
                for i in range(len(self.ups)):
                    self.conv_conds.append(weight_norm(
                        nn.Conv1d(global_channels, base_channels // (2**(i + 1)), 1))
                    )
                self.conv_conds.apply(init_weights)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def output_size(self):
        return self.out_channels

    def _f02source(self, f0: torch.Tensor) -> torch.Tensor:
        return self.source_module(f0.unsqueeze(1))

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        # x in (B, in_channels, T), f0 in (B, T), g in (B, global_channels, 1)

        s = self._f02source(f0)

        x = self.conv_pre(x)
        if self.global_channels > 0 and g is not None:
            x = x + self.conv_global_cond(g)

        for i in range(self.num_upsamples):
            if self.act_pre_each_up_layer:
                x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if self.cond_in_each_up_layer and g is not None:
                x = x + self.conv_conds[i](g)

            # fusion
            x = x + self.source_downs[i](s)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        if self.global_channels > 0:
            remove_weight_norm(self.conv_global_cond)
        if self.cond_in_each_up_layer:
            for l in self.conv_conds:
                remove_weight_norm(l)
        self.source_module.remove_weight_norm()
        for l in self.source_downs:
            remove_weight_norm(l)


class HiFTGenerator(nn.Module):
    """
    HiFTNet Generator: Neural Source Filter + ISTFTNet
    https://arxiv.org/abs/2309.09493
    """
    def __init__(
            self,
            in_channels: int = 80,
            base_channels: int = 512,
            global_channels: int = -1,
            nb_harmonics: int = 8,
            sampling_rate: int = 22050,
            nsf_alpha: float = 0.1,
            nsf_sigma: float = 0.003,
            nsf_voiced_threshold: float = 10,
            upsample_rates: tp.List[int] = [8, 8],
            upsample_kernel_sizes: tp.List[int] = [16, 16],
            istft_params: tp.Dict[str, int] = {"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes: tp.List[int] = [3, 7, 11],
            resblock_dilation_sizes: tp.List[tp.List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            resblock_nonlinear_activation: str = "Snake",
            resblock_nonlinear_activation_params: tp.Dict[str, tp.Any] = {"alpha_logscale": False},
            source_resblock_kernel_sizes: tp.List[int] = [7, 11],
            source_resblock_dilation_sizes: tp.List[tp.List[int]] = [[1, 3, 5], [1, 3, 5]],
            source_resblock_nonlinear_activation: str = "Snake",
            source_resblock_nonlinear_activation_params: tp.Dict[str, tp.Any] = {"alpha_logscale": False},
            use_additional_convs: bool = True,
            cond_in_each_up_layer: bool = False,
            lrelu_slope: float = 0.1,
            act_pre_each_up_layer: bool = True,
            audio_limit: float = 0.99,
    ):
        super(HiFTGenerator, self).__init__()

        self.out_channels = 1
        self.global_channels = global_channels
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.use_additional_convs = use_additional_convs
        self.cond_in_each_up_layer = cond_in_each_up_layer if global_channels > 0 else False
        self.lrelu_slope = lrelu_slope
        self.act_pre_each_up_layer = act_pre_each_up_layer
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * istft_params["hop_len"])

        self.conv_pre = weight_norm(
            Conv1d(in_channels, base_channels, 7, 1, padding=3)
        )

        # Up
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        base_channels // (2**i),
                        base_channels // (2**(i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # Down
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes,
                                          source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(
                    Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1)
                )
            else:
                self.source_downs.append(
                    Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u*2, u, padding=(u//2))
                )

            self.source_resblocks.append(
                ResBlock(base_channels // (2 ** (i + 1)), k, d,
                         use_additional_convs, source_resblock_nonlinear_activation,
                         source_resblock_nonlinear_activation_params)
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2**(i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d, use_additional_convs,
                                               resblock_nonlinear_activation,
                                               resblock_nonlinear_activation_params))

        if self.global_channels > 0:
            self.conv_global_cond = weight_norm(
                Conv1d(global_channels, base_channels, 1)
            )
            self.conv_global_cond.apply(init_weights)

            if self.cond_in_each_up_layer:
                self.conv_conds = nn.ModuleList()
                for i in range(len(self.ups)):
                    self.conv_conds.append(weight_norm(
                        nn.Conv1d(global_channels, base_channels // (2**(i + 1)), 1))
                    )
                self.conv_conds.apply(init_weights)

        self.conv_post = weight_norm(Conv1d(ch, istft_params["n_fft"] + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        window = torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.register_buffer("stft_window", window)

    def output_size(self):
        return self.out_channels

    def _f02source(self, f0: torch.Tensor) -> torch.Tensor:
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t

        har_source, _, _ = self.m_source(f0)
        return har_source.transpose(1, 2)

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        # x in (B, in_channels, T), f0 in (B, T), g in (B, global_channels, 1)

        s = self._f02source(f0)

        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        x = self.conv_pre(x)
        if self.global_channels > 0 and g is not None:
            x = x + self.conv_global_cond(g)

        for i in range(self.num_upsamples):
            if self.act_pre_each_up_layer:
                x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if self.cond_in_each_up_layer and g is not None:
                x = x + self.conv_conds[i](g)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1:, :])  # actually, sin is redundancy

        x = self._istft(magnitude, phase)
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        if self.global_channels > 0:
            remove_weight_norm(self.conv_global_cond)
        if self.cond_in_each_up_layer:
            for l in self.conv_conds:
                remove_weight_norm(l)
        self.source_module.remove_weight_norm()
        for l in self.source_downs:
            remove_weight_norm(l)
        for l in self.source_resblocks:
            l.remove_weight_norm()

    def _stft(self, x):
        spec = torch.stft(
            x,
            self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"], window=self.stft_window,
            return_complex=True)
        spec = torch.view_as_real(spec) # [B, F, TT, 2]
        return spec[...,0], spec[...,1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(
            torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1),
            self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"], window=self.stft_window)

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation
