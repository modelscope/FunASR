"""Positional Encoding Module."""

import math

import torch
import torch.nn as nn
from funasr.modules.embedding import (
    LegacyRelPositionalEncoding, PositionalEncoding, RelPositionalEncoding,
    ScaledPositionalEncoding, StreamPositionalEncoding)
from funasr.modules.subsampling import (
    Conv2dSubsampling, Conv2dSubsampling2, Conv2dSubsampling6,
    Conv2dSubsampling8)
from funasr.modules.subsampling_without_posenc import \
    Conv2dSubsamplingWOPosEnc

from funasr.export.models.language_models.subsampling import (
    OnnxConv2dSubsampling, OnnxConv2dSubsampling2, OnnxConv2dSubsampling6,
    OnnxConv2dSubsampling8)


def get_pos_emb(pos_emb, max_seq_len=512, use_cache=True):
    if isinstance(pos_emb, LegacyRelPositionalEncoding):
        return OnnxLegacyRelPositionalEncoding(pos_emb, max_seq_len, use_cache)
    elif isinstance(pos_emb, ScaledPositionalEncoding):
        return OnnxScaledPositionalEncoding(pos_emb, max_seq_len, use_cache)
    elif isinstance(pos_emb, RelPositionalEncoding):
        return OnnxRelPositionalEncoding(pos_emb, max_seq_len, use_cache)
    elif isinstance(pos_emb, PositionalEncoding):
        return OnnxPositionalEncoding(pos_emb, max_seq_len, use_cache)
    elif isinstance(pos_emb, StreamPositionalEncoding):
        return OnnxStreamPositionalEncoding(pos_emb, max_seq_len, use_cache)
    elif (isinstance(pos_emb, nn.Sequential) and len(pos_emb) == 0) or (
        isinstance(pos_emb, Conv2dSubsamplingWOPosEnc)
    ):
        return pos_emb
    else:
        raise ValueError("Embedding model is not supported.")


class Embedding(nn.Module):
    def __init__(self, model, max_seq_len=512, use_cache=True):
        super().__init__()
        self.model = model
        if not isinstance(model, nn.Embedding):
            if isinstance(model, Conv2dSubsampling):
                self.model = OnnxConv2dSubsampling(model)
                self.model.out[-1] = get_pos_emb(model.out[-1], max_seq_len)
            elif isinstance(model, Conv2dSubsampling2):
                self.model = OnnxConv2dSubsampling2(model)
                self.model.out[-1] = get_pos_emb(model.out[-1], max_seq_len)
            elif isinstance(model, Conv2dSubsampling6):
                self.model = OnnxConv2dSubsampling6(model)
                self.model.out[-1] = get_pos_emb(model.out[-1], max_seq_len)
            elif isinstance(model, Conv2dSubsampling8):
                self.model = OnnxConv2dSubsampling8(model)
                self.model.out[-1] = get_pos_emb(model.out[-1], max_seq_len)
            else:
                self.model[-1] = get_pos_emb(model[-1], max_seq_len)

    def forward(self, x, mask=None):
        if mask is None:
            return self.model(x)
        else:
            return self.model(x, mask)


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class OnnxPositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_seq_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, model, max_seq_len=512, reverse=False, use_cache=True):
        """Construct an PositionalEncoding object."""
        super(OnnxPositionalEncoding, self).__init__()
        self.d_model = model.d_model
        self.reverse = reverse
        self.max_seq_len = max_seq_len
        self.xscale = math.sqrt(self.d_model)
        self._register_load_state_dict_pre_hook(_pre_hook)
        self.pe = model.pe
        self.use_cache = use_cache
        self.model = model
        if self.use_cache:
            self.extend_pe()
        else:
            self.div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )

    def extend_pe(self):
        """Reset the positional encodings."""
        pe_length = len(self.pe[0])
        if self.max_seq_len < pe_length:
            self.pe = self.pe[:, : self.max_seq_len]
        else:
            self.model.extend_pe(torch.tensor(0.0).expand(1, self.max_seq_len))
            self.pe = self.model.pe

    def _add_pe(self, x):
        """Computes positional encoding"""
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)

        x = x * self.xscale
        x[:, :, 0::2] += torch.sin(position * self.div_term)
        x[:, :, 1::2] += torch.cos(position * self.div_term)
        return x

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        if self.use_cache:
            x = x * self.xscale + self.pe[:, : x.size(1)]
        else:
            x = self._add_pe(x)
        return x


class OnnxScaledPositionalEncoding(OnnxPositionalEncoding):
    """Scaled positional encoding module.

    See Sec. 3.2  https://arxiv.org/abs/1809.08895

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_seq_len (int): Maximum input length.

    """

    def __init__(self, model, max_seq_len=512, use_cache=True):
        """Initialize class."""
        super().__init__(model, max_seq_len, use_cache=use_cache)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def _add_pe(self, x):
        """Computes positional encoding"""
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)

        x = x * self.alpha
        x[:, :, 0::2] += torch.sin(position * self.div_term)
        x[:, :, 1::2] += torch.cos(position * self.div_term)
        return x

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        if self.use_cache:
            x = x + self.alpha * self.pe[:, : x.size(1)]
        else:
            x = self._add_pe(x)
        return x


class OnnxLegacyRelPositionalEncoding(OnnxPositionalEncoding):
    """Relative positional encoding module (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_seq_len (int): Maximum input length.

    """

    def __init__(self, model, max_seq_len=512, use_cache=True):
        """Initialize class."""
        super().__init__(model, max_seq_len, reverse=True, use_cache=use_cache)

    def _get_pe(self, x):
        """Computes positional encoding"""
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)

        pe = torch.zeros(x.shape)
        pe[:, :, 0::2] += torch.sin(position * self.div_term)
        pe[:, :, 1::2] += torch.cos(position * self.div_term)
        return pe

    def forward(self, x):
        """Compute positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).

        """
        x = x * self.xscale
        if self.use_cache:
            pos_emb = self.pe[:, : x.size(1)]
        else:
            pos_emb = self._get_pe(x)
        return x, pos_emb


class OnnxRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_seq_len (int): Maximum input length.
    """

    def __init__(self, model, max_seq_len=512, use_cache=True):
        """Construct an PositionalEncoding object."""
        super(OnnxRelPositionalEncoding, self).__init__()
        self.d_model = model.d_model
        self.xscale = math.sqrt(self.d_model)
        self.pe = None
        self.use_cache = use_cache
        if self.use_cache:
            self.extend_pe(torch.tensor(0.0).expand(1, max_seq_len))
        else:
            self.div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None and self.pe.size(1) >= x.size(1) * 2 - 1:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.dtype != x.dtype or self.pe.device != x.device:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def _get_pe(self, x):
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        theta = (
            torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1) * self.div_term
        )
        pe_positive[:, 0::2] = torch.sin(theta)
        pe_positive[:, 1::2] = torch.cos(theta)
        pe_negative[:, 0::2] = -1 * torch.sin(theta)
        pe_negative[:, 1::2] = torch.cos(theta)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        return torch.cat([pe_positive, pe_negative], dim=1)

    def forward(self, x: torch.Tensor, use_cache=True):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        x = x * self.xscale
        if self.use_cache:
            pos_emb = self.pe[
                :,
                self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
            ]
        else:
            pos_emb = self._get_pe(x)
        return x, pos_emb


class OnnxStreamPositionalEncoding(torch.nn.Module):
    """Streaming Positional encoding."""

    def __init__(self, model, max_seq_len=5000, use_cache=True):
        """Construct an PositionalEncoding object."""
        super(StreamPositionalEncoding, self).__init__()
        self.use_cache = use_cache
        self.d_model = model.d_model
        self.xscale = model.xscale
        self.pe = model.pe
        self.use_cache = use_cache
        self.max_seq_len = max_seq_len
        if self.use_cache:
            self.extend_pe()
        else:
            self.div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self):
        """Reset the positional encodings."""
        pe_length = len(self.pe[0])
        if self.max_seq_len < pe_length:
            self.pe = self.pe[:, : self.max_seq_len]
        else:
            self.model.extend_pe(self.max_seq_len)
            self.pe = self.model.pe

    def _add_pe(self, x, start_idx):
        position = torch.arange(start_idx, x.size(1), dtype=torch.float32).unsqueeze(1)
        x = x * self.xscale
        x[:, :, 0::2] += torch.sin(position * self.div_term)
        x[:, :, 1::2] += torch.cos(position * self.div_term)
        return x

    def forward(self, x: torch.Tensor, start_idx: int = 0):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        if self.use_cache:
            return x * self.xscale + self.pe[:, start_idx : start_idx + x.size(1)]
        else:
            return self._add_pe(x, start_idx)
