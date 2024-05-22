import copy
from typing import Optional
from typing import Tuple
from typing import Union
import logging
import numpy as np
import torch
import torch.nn as nn

try:
    from torch_complex.tensor import ComplexTensor
except:
    print("Please install torch_complex firstly")

from funasr.frontends.utils.log_mel import LogMel
from funasr.frontends.utils.stft import Stft
from funasr.frontends.utils.frontend import Frontend
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.register import tables


@tables.register("frontend_classes", "DefaultFrontend")
class DefaultFrontend(nn.Module):
    """Conventional frontend structure for ASR.
    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = None,
        apply_stft: bool = True,
        use_channel: int = None,
        **kwargs,
    ):
        super().__init__()

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length
        self.fs = fs

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.use_channel = use_channel
        self.frontend_type = "default"

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: Union[torch.Tensor, list]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(input_lengths, list):
            input_lengths = torch.tensor(input_lengths)
        if input.dtype == torch.float64:
            input = input.float()
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(input, input_lengths)
        else:
            input_stft = ComplexTensor(input[..., 0], input[..., 1])
            feats_lens = input_lengths
        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                if self.use_channel is not None:
                    input_stft = input_stft[:, :, self.use_channel, :]
                else:
                    # Select 1ch randomly
                    ch = np.random.randint(input_stft.size(2))
                    input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)

        return input_feats, feats_lens

    def _compute_stft(self, input: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens


class MultiChannelFrontend(nn.Module):
    """Conventional frontend structure for ASR.
    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = None,
        frame_length: int = None,
        frame_shift: int = None,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = None,
        apply_stft: bool = True,
        use_channel: int = None,
        lfr_m: int = 1,
        lfr_n: int = 1,
        cmvn_file: str = None,
        mc: bool = True,
    ):
        super().__init__()
        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        if win_length is None and hop_length is None:
            self.win_length = frame_length * 16
            self.hop_length = frame_shift * 16
        elif frame_length is None and frame_shift is None:
            self.win_length = self.win_length
            self.hop_length = self.hop_length
        else:
            logging.error(
                "Only one of (win_length, hop_length) and (frame_length, frame_shift)" "can be set."
            )
            exit(1)

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.use_channel = use_channel
        self.mc = mc
        if not self.mc:
            if self.use_channel is not None:
                logging.info("use the channel %d" % (self.use_channel))
            else:
                logging.info("random select channel")
            self.cmvn_file = cmvn_file
            if self.cmvn_file is not None:
                mean, std = self._load_cmvn(self.cmvn_file)
                self.register_buffer("mean", torch.from_numpy(mean))
                self.register_buffer("std", torch.from_numpy(std))
        self.frontend_type = "multichannelfrontend"

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        # import pdb;pdb.set_trace()
        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(input, input_lengths)
        else:
            input_stft = ComplexTensor(input[..., 0], input[..., 1])
            feats_lens = input_lengths
        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel(sa_asr)
        if input_stft.dim() == 4 and not self.mc:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                if self.use_channel is not None:
                    input_stft = input_stft[:, :, self.use_channel, :]

                else:
                    # Select 1ch randomly
                    ch = np.random.randint(input_stft.size(2))
                    input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)
        if self.mc:
            # MFCCA
            if input_feats.dim() == 4:
                bt = input_feats.size(0)
                channel_size = input_feats.size(2)
                input_feats = (
                    input_feats.transpose(1, 2).reshape(bt * channel_size, -1, 80).contiguous()
                )
                feats_lens = feats_lens.repeat(1, channel_size).squeeze()
            else:
                channel_size = 1
            return input_feats, feats_lens, channel_size
        else:
            # 6. Apply CMVN
            if self.cmvn_file is not None:
                if feats_lens is None:
                    feats_lens = input_feats.new_full([input_feats.size(0)], input_feats.size(1))
                self.mean = self.mean.to(input_feats.device, input_feats.dtype)
                self.std = self.std.to(input_feats.device, input_feats.dtype)
                mask = make_pad_mask(feats_lens, input_feats, 1)

                if input_feats.requires_grad:
                    input_feats = input_feats + self.mean
                else:
                    input_feats += self.mean
                if input_feats.requires_grad:
                    input_feats = input_feats.masked_fill(mask, 0.0)
                else:
                    input_feats.masked_fill_(mask, 0.0)

                input_feats *= self.std

            return input_feats, feats_lens

    def _compute_stft(self, input: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens

    def _load_cmvn(self, cmvn_file):
        with open(cmvn_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        means_list = []
        vars_list = []
        for i in range(len(lines)):
            line_item = lines[i].split()
            if line_item[0] == "<AddShift>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    add_shift_line = line_item[3 : (len(line_item) - 1)]
                    means_list = list(add_shift_line)
                    continue
            elif line_item[0] == "<Rescale>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    rescale_line = line_item[3 : (len(line_item) - 1)]
                    vars_list = list(rescale_line)
                    continue
        means = np.array(means_list).astype(np.float)
        vars = np.array(vars_list).astype(np.float)
        return means, vars
