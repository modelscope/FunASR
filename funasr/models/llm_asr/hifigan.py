# Copyright 2023 KaiHu
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HIFI-GAN"""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple, List, Union
import typing as tp
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types
from funasr.train_utils.device_funcs import force_gatherable
from librosa.filters import mel as librosa_mel_fn
import logging
from funasr.utils.hinter import hint_once


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
        center=False,
        device='cuda',
        feat_type="power_log",
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length, device=device).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float().to(device)
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.center = center
        self.feat_type = feat_type

    def forward(self, audioin):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audioin, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
        )
        if self.feat_type == "mag_log10":
            power_spec = torch.sqrt(torch.sum(torch.pow(fft, 2), dim=[-1]))
            mel_output = torch.matmul(self.mel_basis, power_spec)
            return torch.log10(torch.clamp(mel_output, min=1e-5))
        power_spec = torch.sum(torch.pow(fft, 2), dim=[-1])
        mel_spec = torch.matmul(self.mel_basis, torch.sqrt(power_spec + 1e-9))
        return self.spectral_normalize(mel_spec)


    @classmethod
    def spectral_normalize(cls, spec, C=1, clip_val=1e-5):
        output = cls.dynamic_range_compression(spec, C, clip_val)
        return output

    @classmethod
    def spectral_de_normalize_torch(cls, spec, C=1, clip_val=1e-5):
        output = cls.dynamic_range_decompression(spec, C, clip_val)
        return output

    @staticmethod
    def dynamic_range_compression(x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)

    @staticmethod
    def dynamic_range_decompression(x, C=1):
        return torch.exp(x) / C


class HifiGan(nn.Module):
    """HIFIGAN-style vocoders (generator [stack of time-level-upsampling blocks] + discriminator).
       NSF-HIFIGAN, HiFTNet Optional.
    """

    def __init__(
            self,
            input_size: int,
            frontend: torch.nn.Module = None,
            nsf_augmented: bool = False,
            f0_predictor: dict = None,
            generator: dict = None,
            discriminator: dict = None,
            target_sample_hz: int = 22_050,
            multi_mel_spectral_window: Union[Tuple, List] = tuple([1024]),
            multi_mel_spectral_hop: Union[Tuple, List] = tuple([256]),
            multi_mel_spectral_fft: Union[Tuple, List] = tuple([1024]),
            multi_mel_spectral_n_mels: Union[Tuple, List] = tuple([80]),
            mel_fmin: float = 0,
            mel_fmax: float = 8000,
            mel_fmax_for_loss: Optional[float] = None,
            multi_mel_spectral_recon_loss_weight: Union[Tuple[float], List[float]] = tuple([45]),
            adversarial_loss_weight: float = 1.0,
            feat_match_loss_weight: float = 2.0,
            tpr_loss_params: tp.Dict[str, tp.Any] = {"weight": 0.0, "tau": 0.04},
            mel_feat_type="power_log",
    ):
        """Initialize HifiGan model.
        Args:
            f0_predictor: f0 predictor (pretrained && frozen) for NSF-HIFIGAN, Optional.
            generator: hifigan generator
            discriminator: several discriminators, such as MSD, MPD, MRD
            multi_mel_spectral_window: stft window length
            multi_mel_spectral_hop: stft hop length
            multi_mel_spectral_fft: fft bins
            multi_mel_spectral_n_mels: Mel frequency bins
            mel_fmin: fmin for mel
            mel_fmax: fmax for mel
            mel_fmax_for_loss: fmax for multi mel spectral loss
            multi_mel_spectral_recon_loss_weight: the weight of frequency-domain reconstruction loss
            adversarial_loss_weight: the weight of adversarial loss from discriminator
            feat_match_loss_weight: the weight of intermediate feature loss from discriminator
            tpr_loss_params: the weight and tau of Truncated Pointwise Relativistic (TPR) loss from discriminator.
        """
        super().__init__()

        self.decoder = self.build_decoder(generator)
        # Used by task and trainer
        self.gen_model_list = [self.decoder]

        # nsf-hifigan or original hifigan
        self.nsf_augmented = nsf_augmented
        if nsf_augmented:
            assert f0_predictor is not None
            self.f0_predictor = self.build_f0_predictor(f0_predictor)
            # frozen
            for param in self.f0_predictor.parameters():
                param.requires_grad = False
            self.gen_model_list.append(self.f0_predictor)

        self.discriminator = self.build_discriminator(discriminator)

        self.multi_mel_spec_transforms = nn.ModuleList()
        for n_fft, hop_len, win_len, n_mel in zip(multi_mel_spectral_fft, multi_mel_spectral_hop,
                                                  multi_mel_spectral_window, multi_mel_spectral_n_mels):
            self.multi_mel_spec_transforms.append(
                Audio2Mel(
                    n_fft=n_fft,
                    hop_length=hop_len,
                    win_length=win_len,
                    sampling_rate=target_sample_hz,
                    n_mel_channels=n_mel,
                    mel_fmin=mel_fmin,
                    mel_fmax=mel_fmax_for_loss,
                    center=False,
                )
            )

        self.mel_spec_transform = Audio2Mel(
                n_fft=multi_mel_spectral_fft[0],
                hop_length=multi_mel_spectral_hop[0],
                win_length=multi_mel_spectral_window[0],
                sampling_rate=target_sample_hz,
                n_mel_channels=multi_mel_spectral_n_mels[0],
                mel_fmin=mel_fmin,
                mel_fmax=mel_fmax,
                center=False,
                feat_type=mel_feat_type,
        )

        # loss weights
        self.multi_mel_spectral_recon_loss_weight = multi_mel_spectral_recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feat_match_loss_weight = feat_match_loss_weight
        self.tpr_loss_weight = tpr_loss_params.get("weight", 0.0)
        self.tpr_loss_tau = tpr_loss_params.get("tau", 0.04)
        self.register_buffer('zero', torch.tensor([0.]), persistent=False)
        self.gen_loss = 0
        self.sample_rate = target_sample_hz
        self.forward_step = 0

    def build_decoder(self, conf):
        from funasr.models.llm_asr.hifigan_module.generator import HiFTGenerator
        return HiFTGenerator(**conf)

    def build_f0_predictor(self, conf):
        from funasr.models.llm_asr.hifigan_module.nsf_utils import ConvRNNF0Predictor
        return ConvRNNF0Predictor(**conf)

    def build_discriminator(self, conf):
        from funasr.models.llm_asr.hifigan_module.discriminator import MultipleDiscriminator
        return MultipleDiscriminator(**conf)

    @property
    def generator(self):
        return torch.nn.ModuleList(self.gen_model_list)

    def forward(
        self,
        forward_generator: bool = True,
        batch: Dict = None,
    ) -> Dict[str, Any]:
        """Forward functions of generator and discriminator.

        Args:
            forward_generator (bool): Whether to forward generator.
            batch (Dict[str, Tensor]): one batch including:
                speech (Tensor): Speech waveform tensor (B, T_wav).
                speech_lengths (Tensor): Speech length tensor (B,).

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        if forward_generator:
            if self.training:
                self.forward_step += 1
            return self._forward_generator(
                speech=batch["speech"],
                speech_lengths=batch["speech_lengths"],
            )
        else:
            return self._forward_discriminator(
                speech=batch["speech"],
                speech_lengths=batch["speech_lengths"],
            )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Given a tensor `x`, returns the encoded representation for `x`
        """
        assert x.dim() == 3
        _, channel, length = x.size()
        assert channel == 1
        mel = self.mel_spec_transform(x)
        return mel.squeeze()

    @torch.no_grad()
    def _f0_pred(self, x: torch.Tensor) -> torch.Tensor:
        """Given a tensor `x`, return the predicted f0 for `x`, x in (B, C, T)
        """
        if self.nsf_augmented:
            f0 = self.f0_predictor(x)
            if len(f0.shape) == 1:
                f0 = f0.unsqueeze(0)
            return f0
        else:
            return torch.zeros_like(x)

    def _decode(self, x: torch.Tensor, g: Union[torch.Tensor] = None) -> torch.Tensor:
        """Decode the given representation into a waveform.

        Args:
            x (Tensor): Speech representation tensor (B, C1, T)
            g (Tensor): Global conditional vector (B, C2, 1).
        """
        if self.nsf_augmented:
            f0 = self._f0_pred(x)
            return self.decoder(x, f0, g)
        else:
            return self.decoder(x, g)

    def _forward_generator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = speech.size(0)
        speech = speech.unsqueeze(1)
        orig_speech = speech.clone()

        mel = self._encode(speech) # [B, C, T]
        recon_speech = self._decode(mel)[:, :, :speech.shape[-1]]

        # L1 Mel-Spectrogram Loss
        multi_mel_recon_loss = self.zero
        for lamda, mel_transform in zip(self.multi_mel_spectral_recon_loss_weight, self.multi_mel_spec_transforms):
            orig_mel, recon_mel = map(mel_transform, (orig_speech, recon_speech))
            multi_mel_recon_loss = multi_mel_recon_loss + lamda * F.l1_loss(orig_mel, recon_mel)

        # calculate discriminator outputs
        # disc_outputs in the format [disc1_outputs, disc2_outputs, ...]
        # disc1_outputs includes [logits, intermediates]
        # intermediates includes [layer_1_intermediate, layer_2_intermediate, ...]
        fake_disc_outputs = self.discriminator(recon_speech)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            real_disc_outputs = self.discriminator(orig_speech)

        # calculate discriminator loss including adversarial, feat matching losses and tpr losses [Optional]
        adversarial_losses = []
        disc_feature_losses = []
        tpr_losses = []
        for real_output, fake_output in zip(real_disc_outputs, fake_disc_outputs):
            real_logits, real_intermediates = real_output
            fake_logits, fake_intermediates = fake_output
            adversarial_losses.append(torch.mean((1 - fake_logits)**2))
            for real_inter, fake_inter in zip(real_intermediates, fake_intermediates):
                _loss = torch.mean(torch.abs(real_inter.detach() - fake_inter))
                disc_feature_losses.append(_loss)

            if self.tpr_loss_weight > 0.0:
                tau = self.tpr_loss_tau
                m_DG = torch.median((fake_logits - real_logits))
                L_rel = torch.mean((((fake_logits - real_logits) - m_DG) ** 2)[fake_logits < real_logits + m_DG])
                tpr_losses.append(tau - F.relu(tau - L_rel))

        adversarial_loss = torch.stack(adversarial_losses).sum()
        feat_match_loss = torch.stack(disc_feature_losses).sum()
        tpr_loss = torch.zeros_like(adversarial_loss)
        if len(tpr_losses) > 0:
            tpr_loss = torch.stack(tpr_losses).sum()

        # calculate losses
        gen_loss = multi_mel_recon_loss + \
                   adversarial_loss * self.adversarial_loss_weight + \
                   feat_match_loss * self.feat_match_loss_weight + \
                   tpr_loss  * self.tpr_loss_weight
        self.gen_loss += gen_loss.item()
        loss = gen_loss

        stats = dict(
            generator_loss=loss.item(),
            generator_multi_mel_recon_loss=multi_mel_recon_loss.item(),
            generator_adv_loss=adversarial_loss.item(),
            generator_feat_match_loss=feat_match_loss.item(),
            generator_tpr_loss=tpr_loss.item(),
            batch_size=batch_size,
            batch_length=speech.shape[2],
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 0,  # needed for trainer
            "real": orig_speech,
            "fake": recon_speech,
        }

    def _forward_discriminator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Dict[str, Any]:
        """Perform discriminator forward.

        Args:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).
        """
        # setup
        batch_size = speech.size(0)
        speech = speech.unsqueeze(1)
        orig_speech = speech.clone()

        # A: calculate generator outputs
        with torch.no_grad():
            # do not store generator gradient in discriminator turn
            mel = self._encode(speech) # [B, C, T]
            recon_speech = self._decode(mel)[:, :, :speech.shape[-1]]

        # B: calculate discriminator outputs
        real, fake = orig_speech.clone(), recon_speech.detach()
        real_disc_outputs = self.discriminator(real)
        fake_disc_outputs = self.discriminator(fake)

        # C: calculate discriminator losses, tpr losses [Optional]
        disc_losses = []
        tpr_losses = []
        for real_output, fake_output in zip(real_disc_outputs, fake_disc_outputs):
            real_logits, real_intermediates = real_output
            fake_logits, fake_intermediates = fake_output
            one_disc_loss = torch.mean((1-real_logits) ** 2) + torch.mean((0 - fake_logits) ** 2)
            disc_losses.append(one_disc_loss)

            if self.tpr_loss_weight > 0.0:
                tau = self.tpr_loss_tau
                m_DG = torch.median((real_logits - fake_logits))
                L_rel = torch.mean((((real_logits - fake_logits) - m_DG) ** 2)[real_logits < fake_logits + m_DG])
                tpr_losses.append(tau - F.relu(tau - L_rel))

        disc_loss = torch.stack(disc_losses).sum()
        tpr_loss = torch.zeros_like(disc_loss)
        if len(tpr_losses) > 0:
            tpr_loss = torch.stack(tpr_losses).sum()

        self.gen_loss = 0

        loss = disc_loss + self.tpr_loss_weight * tpr_loss

        stats = dict(
            discriminator_total_loss=loss.item(),
            discriminator_loss=disc_loss.item(),
            discriminator_tpr_loss=tpr_loss.item(),
        )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # needed for trainer
            "real": orig_speech,
            "fake": recon_speech,
        }

    def inference(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            x (torch.Tensor): input representation, B x T x C

        Returns:
            Dict[str, Tensor]:
                * recon_speech (Tensor): Reconstructed waveform tensor (B, T_wav).

        """

        recon_speech = self._decode(x.transpose(1, 2)).squeeze(1)
        retval = dict(
            recon_speech=recon_speech,
        )
        return retval

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

    @property
    def input_size(self):
        return
