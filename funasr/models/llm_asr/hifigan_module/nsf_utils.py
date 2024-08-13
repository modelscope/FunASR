"""
Neural Source Filter based modules implementation.

Neural source-filter waveform models for statistical parametric speech synthesis

"""

import numpy as np
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    @torch.no_grad()
    def forward(self, f0):
        """
        :param f0: [B, 1, sample_len], Hz
        :return: [B, 1, sample_len]
        """

        F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1))).to(f0.device)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i:i+1, :] = f0 * (i+1) / self.sampling_rate

        theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        u_dist = Uniform(low=-np.pi, high=np.pi)
        phase_vec = u_dist.sample(sample_shape=(f0.size(0), self.harmonic_num + 1, 1)).to(F_mat.device)
        phase_vec[:, 0, :] = 0

        # generate sine waveforms
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)
        
        # generate uv signal
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise
       

class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x.transpose(1,2))
            sine_wavs = sine_wavs.transpose(1,2)
            uv = uv.transpose(1,2)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv
    

class SourceModule(torch.nn.Module):
    def __init__(self,
                 nb_harmonics: int,
                 upsample_ratio: int,
                 sampling_rate: int,
                 alpha: float = 0.1,
                 sigma: float = 0.003,
                 voiced_threshold: float = 10
                 ):
        super(SourceModule, self).__init__()

        self.nb_harmonics = nb_harmonics
        self.upsample_ratio = upsample_ratio
        self.sampling_rate = sampling_rate
        self.alpha = alpha
        self.sigma = sigma
        self.voiced_threshold = voiced_threshold

        self.ffn = nn.Sequential(
            weight_norm(nn.Conv1d(self.nb_harmonics + 1, 1, kernel_size=1, stride=1)),
            nn.Tanh())

    def f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def forward(self, f0):
        """
        :param f0: [B, 1, frame_len], Hz
        :return: [B, 1, sample_len]
        """
        with torch.no_grad():
            uv = self.f02uv(f0)
            f0_samples = F.interpolate(f0, scale_factor=(self.upsample_ratio), mode='nearest')
            uv_samples = F.interpolate(uv, scale_factor=(self.upsample_ratio), mode='nearest')

            F_mat = torch.zeros((f0_samples.size(0), self.nb_harmonics + 1, f0_samples.size(-1))).to(f0_samples.device)
            for i in range(self.nb_harmonics + 1):
                F_mat[:, i:i+1, :] = f0_samples * (i+1) / self.sampling_rate

            theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
            u_dist = Uniform(low=-np.pi, high=np.pi)
            phase_vec = u_dist.sample(sample_shape=(f0.size(0), self.nb_harmonics + 1, 1)).to(F_mat.device)
            phase_vec[:, 0, :] = 0

            n_dist = Normal(loc=0., scale=self.sigma)
            noise = n_dist.sample(sample_shape=(f0_samples.size(0), self.nb_harmonics + 1, f0_samples.size(-1))).to(F_mat.device)

            e_voice = self.alpha * torch.sin(theta_mat + phase_vec) + noise
            e_unvoice = self.alpha / 3 / self.sigma * noise

            e = e_voice * uv_samples + e_unvoice * (1 - uv_samples)

        return self.ffn(e)

    def remove_weight_norm(self):
        remove_weight_norm(self.ffn[0])


class ConvRNNF0Predictor(nn.Module):
    def __init__(self,
                 num_class: int = 1,
                 in_channels: int = 80,
                 cond_channels: int = 512,
                 use_cond_rnn: bool = True,
                 bidirectional_rnn: bool = False,
                 ):

        super().__init__()

        self.num_class = num_class
        self.use_cond_rnn = use_cond_rnn

        self.condnet = nn.Sequential(
            weight_norm(
                nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
        )

        if self.use_cond_rnn:
            self.rnn = nn.GRU(
                cond_channels,
                cond_channels // 2 if bidirectional_rnn else cond_channels,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional_rnn,
            )

        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.condnet(x)
        if self.use_cond_rnn:
            x, _ = self.rnn(x.transpose(1, 2))
        else:
            x = x.transpose(1, 2)

        return torch.abs(self.classifier(x).squeeze(-1))



