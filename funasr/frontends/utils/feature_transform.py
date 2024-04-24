from typing import List
from typing import Tuple
from typing import Union

import librosa
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor

from funasr.models.transformer.utils.nets_utils import make_pad_mask


class FeatureTransform(torch.nn.Module):
    def __init__(
        self,
        # Mel options,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = None,
        # Normalization
        stats_file: str = None,
        apply_uttmvn: bool = True,
        uttmvn_norm_means: bool = True,
        uttmvn_norm_vars: bool = False,
    ):
        super().__init__()
        self.apply_uttmvn = apply_uttmvn

        self.logmel = LogMel(fs=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.stats_file = stats_file
        if stats_file is not None:
            self.global_mvn = GlobalMVN(stats_file)
        else:
            self.global_mvn = None

        if self.apply_uttmvn is not None:
            self.uttmvn = UtteranceMVN(norm_means=uttmvn_norm_means, norm_vars=uttmvn_norm_vars)
        else:
            self.uttmvn = None

    def forward(
        self, x: ComplexTensor, ilens: Union[torch.LongTensor, np.ndarray, List[int]]
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        # (B, T, F) or (B, T, C, F)
        if x.dim() not in (3, 4):
            raise ValueError(f"Input dim must be 3 or 4: {x.dim()}")
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(np.asarray(ilens)).to(x.device)

        if x.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(x.size(2))
                h = x[:, :, ch, :]
            else:
                # Use the first channel
                h = x[:, :, 0, :]
        else:
            h = x

        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        h = h.real**2 + h.imag**2

        h, _ = self.logmel(h, ilens)
        if self.stats_file is not None:
            h, _ = self.global_mvn(h, ilens)
        if self.apply_uttmvn:
            h, _ = self.uttmvn(h, ilens)

        return h, ilens


class LogMel(torch.nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.mel

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
        norm: {None, 1, np.inf} [scalar]
            if 1, divide the triangular mel weights by the width of the mel band
            (area normalization).  Otherwise, leave all the triangles aiming for
            a peak value of 1.0

    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = None,
        htk: bool = False,
        norm=1,
    ):
        super().__init__()

        _mel_options = dict(
            sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm
        )
        self.mel_options = _mel_options

        # Note(kamo): The mel matrix of librosa is different from kaldi.
        melmat = librosa.filters.mel(**_mel_options)
        # melmat: (D2, D1) -> (D1, D2)
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
        self, feat: torch.Tensor, ilens: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = torch.matmul(feat, self.melmat)

        logmel_feat = (mel_feat + 1e-20).log()
        # Zero padding
        logmel_feat = logmel_feat.masked_fill(make_pad_mask(ilens, logmel_feat, 1), 0.0)
        return logmel_feat, ilens


class GlobalMVN(torch.nn.Module):
    """Apply global mean and variance normalization

    Args:
        stats_file(str): npy file of 1-dim array or text file.
            From the _first element to
            the {(len(array) - 1) / 2}th element are treated as
            the sum of features,
            and the rest excluding the last elements are
            treated as the sum of the square value of features,
            and the last elements eqauls to the number of samples.
        std_floor(float):
    """

    def __init__(
        self,
        stats_file: str,
        norm_means: bool = True,
        norm_vars: bool = True,
        eps: float = 1.0e-20,
    ):
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars

        self.stats_file = stats_file
        stats = np.load(stats_file)

        stats = stats.astype(float)
        assert (len(stats) - 1) % 2 == 0, stats.shape

        count = stats.flatten()[-1]
        mean = stats[: (len(stats) - 1) // 2] / count
        var = stats[(len(stats) - 1) // 2 : -1] / count - mean * mean
        std = np.maximum(np.sqrt(var), eps)

        self.register_buffer("bias", torch.from_numpy(-mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(1 / std.astype(np.float32)))

    def extra_repr(self):
        return (
            f"stats_file={self.stats_file}, "
            f"norm_means={self.norm_means}, norm_vars={self.norm_vars}"
        )

    def forward(
        self, x: torch.Tensor, ilens: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        # feat: (B, T, D)
        if self.norm_means:
            x += self.bias.type_as(x)
            x.masked_fill(make_pad_mask(ilens, x, 1), 0.0)

        if self.norm_vars:
            x *= self.scale.type_as(x)
        return x, ilens


class UtteranceMVN(torch.nn.Module):
    def __init__(self, norm_means: bool = True, norm_vars: bool = False, eps: float = 1.0e-20):
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps

    def extra_repr(self):
        return f"norm_means={self.norm_means}, norm_vars={self.norm_vars}"

    def forward(
        self, x: torch.Tensor, ilens: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        return utterance_mvn(
            x, ilens, norm_means=self.norm_means, norm_vars=self.norm_vars, eps=self.eps
        )


def utterance_mvn(
    x: torch.Tensor,
    ilens: torch.LongTensor,
    norm_means: bool = True,
    norm_vars: bool = False,
    eps: float = 1.0e-20,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Apply utterance mean and variance normalization

    Args:
        x: (B, T, D), assumed zero padded
        ilens: (B, T, D)
        norm_means:
        norm_vars:
        eps:

    """
    ilens_ = ilens.type_as(x)
    # mean: (B, D)
    mean = x.sum(dim=1) / ilens_[:, None]

    if norm_means:
        x -= mean[:, None, :]
        x_ = x
    else:
        x_ = x - mean[:, None, :]

    # Zero padding
    x_.masked_fill(make_pad_mask(ilens, x_, 1), 0.0)
    if norm_vars:
        var = x_.pow(2).sum(dim=1) / ilens_[:, None]
        var = torch.clamp(var, min=eps)
        x /= var.sqrt()[:, None, :]
        x_ = x
    return x_, ilens


def feature_transform_for(args, n_fft):
    return FeatureTransform(
        # Mel options,
        fs=args.fbank_fs,
        n_fft=n_fft,
        n_mels=args.n_mels,
        fmin=args.fbank_fmin,
        fmax=args.fbank_fmax,
        # Normalization
        stats_file=args.stats_file,
        apply_uttmvn=args.apply_uttmvn,
        uttmvn_norm_means=args.uttmvn_norm_means,
        uttmvn_norm_vars=args.uttmvn_norm_vars,
    )
