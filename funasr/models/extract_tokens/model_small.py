import logging
import time
import kaldiio, os
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Iterable, Optional

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.search import Hypothesis
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.metrics.compute_acc import compute_accuracy, th_accuracy
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.utils.hinter import hint_once


class SinusoidalPositionEncoder(torch.nn.Module):
    """ """

    def __int__(self, d_model=80, dropout_rate=0.1):
        pass

    def encode(
        self, positions: torch.Tensor = None, depth: int = None, dtype: torch.dtype = torch.float32
    ):
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (
            depth / 2 - 1
        )
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x):
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)

        return x + position_encoding


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MultiHeadedAttentionSANM(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        n_head,
        in_feat,
        n_feat,
        dropout_rate,
        kernel_size,
        sanm_shfit=0,
        lora_list=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        # self.linear_q = nn.Linear(n_feat, n_feat)
        # self.linear_k = nn.Linear(n_feat, n_feat)
        # self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time1, d_k)
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)

        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder

            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            min_value = -float(
                "inf"
            )  # float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        return att_outs + fsmn_memory

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        if chunk_size is not None and look_back > 0 or look_back == -1:
            if cache is not None:
                k_h_stride = k_h[:, :, : -(chunk_size[2]), :]
                v_h_stride = v_h[:, :, : -(chunk_size[2]), :]
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)

                cache["k"] = torch.cat((cache["k"], k_h_stride), dim=2)
                cache["v"] = torch.cat((cache["v"], v_h_stride), dim=2)
                if look_back != -1:
                    cache["k"] = cache["k"][:, :, -(look_back * chunk_size[1]) :, :]
                    cache["v"] = cache["v"][:, :, -(look_back * chunk_size[1]) :, :]
            else:
                cache_tmp = {
                    "k": k_h[:, :, : -(chunk_size[2]), :],
                    "v": v_h[:, :, : -(chunk_size[2]), :],
                }
                cache = cache_tmp
        fsmn_memory = self.forward_fsmn(v, None)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, None)
        return att_outs + fsmn_memory, cache


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()

    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayerSANM, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.in_size == self.size:
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            x = residual + attn
        else:
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)

        return x, cache


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask, is_pad_mask=is_pad_mask)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            if not is_pad_mask:
                qk = qk + mask[:n_ctx, :n_ctx]
            else:
                mask = mask.unsqueeze(1).eq(0)  # (batch, 1, t, 1)
                min_value = -float(
                    "inf"
                )  # min_value = float(np.finfo(torch.tensor(0, dtype=qk.dtype).numpy().dtype).min)
                qk = qk.masked_fill(mask, min_value)

        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        if mask is not None and is_pad_mask:
            w = w.masked_fill(mask, 0.0)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class MultiHeadAttentionSdpa(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask, is_pad_mask=is_pad_mask, is_causal=False)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)
        is_causal = kwargs.get("is_causal", False)
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.5
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if mask is not None:
            if not is_pad_mask:
                mask = None
                is_causal = True
            else:
                mask = mask.unsqueeze(1).to(torch.bool)  # (batch, 1, 1, t)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )
        if mask is not None:
            attn_output = attn_output.masked_fill(mask.transpose(2, 3).logical_not(), 0.0)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(start_dim=2)
        return attn_output, None


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(
            self.inv_freq
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()
        self.attention_heads = attention_heads
        self.query = Linear(linear_units, linear_units)
        self.key = Linear(linear_units, linear_units, bias=False)
        self.value = Linear(linear_units, linear_units)
        self.out = Linear(linear_units, linear_units)
        self.rotary_emb = RotaryEmbedding(
            linear_units // attention_heads,
            max_position_embeddings=kwargs.get("max_position_embeddings", 2048),
            base=kwargs.get("rope_theta", 10000),
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        position_ids = kwargs.get("position_ids", None)
        kv_seq_len = v.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        qk = q @ k
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, t, 1)
            min_value = -float(
                "inf"
            )  # min_value = float(np.finfo(torch.tensor(0, dtype=qk.dtype).numpy().dtype).min)
            qk = qk.masked_fill(mask, min_value)

        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        if mask is not None:
            w = w.masked_fill(mask, 0.0)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class MultiHeadAttentionSdpaRoPE(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()
        self.attention_heads = attention_heads
        self.query = Linear(linear_units, linear_units)
        self.key = Linear(linear_units, linear_units, bias=False)
        self.value = Linear(linear_units, linear_units)
        self.out = Linear(linear_units, linear_units)
        self.rotary_emb = RotaryEmbedding(
            linear_units // attention_heads,
            max_position_embeddings=kwargs.get("max_position_embeddings", 2048),
            base=kwargs.get("rope_theta", 10000),
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        is_causal = kwargs.get("is_causal", False)
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.5
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        position_ids = kwargs.get("position_ids", None)
        kv_seq_len = v.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if mask is not None:
            mask = mask.unsqueeze(1).to(torch.bool)  # (batch, 1, 1, t)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )
        if mask is not None:
            attn_output = attn_output.masked_fill(mask.transpose(2, 3).logical_not(), 0.0)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(start_dim=2)
        return attn_output, None


class MultiHeadAttentionFSMNRoPE(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()
        self.attention_heads = attention_heads
        self.query = Linear(linear_units, linear_units)
        self.key = Linear(linear_units, linear_units, bias=False)
        self.value = Linear(linear_units, linear_units)
        self.out = Linear(linear_units, linear_units)
        self.rotary_emb = RotaryEmbedding(
            linear_units // attention_heads,
            max_position_embeddings=kwargs.get("max_position_embeddings", 2048),
            base=kwargs.get("rope_theta", 10000),
        )

        self.fsmn_block = nn.Conv1d(
            linear_units,
            linear_units,
            kwargs.get("kernel_size", 15),
            stride=1,
            padding=0,
            groups=linear_units,
            bias=False,
        )
        # padding
        left_padding = (kwargs.get("kernel_size", 15) - 1) // 2
        left_padding = left_padding + kwargs.get("sanm_shfit", 0)
        right_padding = kwargs.get("kernel_size", 15) - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.dropout = torch.nn.Dropout(kwargs.get("dropout_rate", 0.0))

    def fsmn(self, inputs, mask):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2) + inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        memory = self.fsmn(v, mask=mask)
        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv) + memory, qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        b, t, d = q.shape
        scale = (d // self.attention_heads) ** -0.5
        q = q.view(*q.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)

        position_ids = kwargs.get("position_ids", None)
        kv_seq_len = v.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        qk = q @ k
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, t, 1)
            min_value = -float(
                "inf"
            )  # min_value = float(np.finfo(torch.tensor(0, dtype=qk.dtype).numpy().dtype).min)
            qk = qk.masked_fill(mask, min_value)

        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        if mask is not None:
            w = w.masked_fill(mask, 0.0)

        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class MultiHeadAttentionFSMNSdpaRoPE(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()

        self.attention_heads = attention_heads
        self.query = Linear(linear_units, linear_units)
        self.key = Linear(linear_units, linear_units, bias=False)
        self.value = Linear(linear_units, linear_units)
        self.out = Linear(linear_units, linear_units)
        self.rotary_emb = RotaryEmbedding(
            linear_units // attention_heads,
            max_position_embeddings=kwargs.get("max_position_embeddings", 2048),
            base=kwargs.get("rope_theta", 10000),
        )

        self.fsmn_block = nn.Conv1d(
            linear_units,
            linear_units,
            kwargs.get("kernel_size", 15),
            stride=1,
            padding=0,
            groups=linear_units,
            bias=False,
        )
        # padding
        left_padding = (kwargs.get("kernel_size", 15) - 1) // 2
        left_padding = left_padding + kwargs.get("sanm_shfit", 0)
        right_padding = kwargs.get("kernel_size", 15) - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.dropout = torch.nn.Dropout(kwargs.get("dropout_rate", 0.0))

    def fsmn(self, inputs, mask):
        b, t, d = inputs.size()  # b, t, d
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))  # b, t, 1
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2) + inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        memory = self.fsmn(v, mask=mask)

        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv) + memory, qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):
        is_causal = kwargs.get("is_causal", False)
        b, t, d = q.shape
        scale = (d // self.attention_heads) ** -0.5
        q = q.view(*q.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)

        position_ids = kwargs.get("position_ids", None)
        kv_seq_len = v.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if mask is not None:
            mask = mask.unsqueeze(1).to(torch.bool)  # (batch, 1, 1, t)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )
        if mask is not None:
            attn_output = attn_output.masked_fill(mask.transpose(2, 3).logical_not(), 0.0)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(start_dim=2)
        return attn_output, None


att_type_dict = {
    "default": MultiHeadAttention,
    "sdpa": MultiHeadAttentionSdpa,
    "self_att": MultiHeadAttentionRoPE,
    "self_att_sdpa": MultiHeadAttentionSdpaRoPE,
    "self_att_fsmn": MultiHeadAttentionFSMNRoPE,
    "self_att_fsmn_sdpa": MultiHeadAttentionFSMNSdpaRoPE,
}


class EncoderLayerSANMLarge(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()

        att_type = kwargs.get("att_type", "self_att_fsmn_sdpa")
        self.attn = att_type_dict[att_type](linear_units, attention_heads, **kwargs)
        self.attn_ln = LayerNorm(linear_units)

        n_mlp = linear_units * 4
        self.mlp = nn.Sequential(
            Linear(linear_units, n_mlp), nn.GELU(), Linear(n_mlp, linear_units)
        )
        self.mlp_ln = LayerNorm(linear_units)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        x = x + self.attn(self.attn_ln(x), mask=mask, **kwargs)[0]

        x = x + self.mlp(self.mlp_ln(x))
        return x


# @tables.register("encoder_classes", "SenseVoiceEncoder")
class SenseVoiceEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        linear_units: int,
        attention_heads: int,
        num_blocks: int,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = Conv1d(input_size, linear_units, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv1d(linear_units, linear_units, kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList(
            [
                EncoderLayerSANMLarge(linear_units, attention_heads, **kwargs)
                for _ in range(num_blocks)
            ]
        )
        self.ln_post = LayerNorm(linear_units)
        self.use_padmask = kwargs.get("use_padmask", True)
        self.downsample_rate = kwargs.get("downsample_rate", 4)

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor = None,
        **kwargs,
    ):
        use_padmask = self.use_padmask
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = n_frames

        if ilens is not None:
            if self.downsample_rate == 4:
                olens = (
                    1
                    + (ilens - self.conv1.kernel_size[0] + 2 * self.conv1.padding[0])
                    // self.conv1.stride[0]
                )
            else:
                olens = ilens
            olens = (
                1
                + (olens - self.conv2.kernel_size[0] + 2 * self.conv2.padding[0])
                // self.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        if use_padmask and olens is not None:
            padding_mask = (~make_pad_mask(olens)[:, None, :]).to(torch.bool).to(x.device)
        else:
            padding_mask = None

        device = x.device
        seq_length = x.shape[1]
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        for layer, block in enumerate(self.blocks):
            x = block(x, mask=padding_mask, position_ids=position_ids)

        x = self.ln_post(x)

        if ilens is None:
            return x
        else:
            return x, olens


@tables.register("encoder_classes", "SenseVoiceQuantizedEncoder")
class SenseVoiceQuantizedEncoder(SenseVoiceEncoder):
    def __init__(
        self,
        input_size,
        linear_units: int,
        attention_heads: int,
        num_blocks: int,
        quantize_layer_idx: int,
        normalized_quant_input: bool,
        quantizer_config: dict,
        **kwargs,
    ):
        super().__init__(input_size, linear_units, attention_heads, num_blocks, **kwargs)
        self.linear_units = linear_units
        self.quantize_layer_idx = quantize_layer_idx
        self.normalized_quant_input = normalized_quant_input
        self.quantizer = self.build_quantizer(quantizer_config)

    def build_quantizer(self, vq_config):
        if vq_config is None:
            return None
        name = vq_config.pop("name", "costume_quantizer")
        if name == "costume_quantizer":
            from funasr.models.sense_voice.quantizer.costume_quantizer import CostumeQuantizer

            quantizer = CostumeQuantizer(
                input_size=self.linear_units,
                **vq_config,
            )
            vq_config["name"] = "costume_quantizer"
            return quantizer
        elif name == "lookup_free_quantizer":
            from funasr.models.sense_voice.quantizer.lookup_free_quantizer import LFQ

            quantizer = LFQ(
                input_size=self.linear_units,
                **vq_config,
            )
            vq_config["name"] = "lookup_free_quantizer"
            return quantizer
        elif name == "finite_scalar_quantizer":
            from funasr.models.sense_voice.quantizer.finite_scalar_quantizer import FSQ

            quantizer = FSQ(
                input_size=self.linear_units,
                **vq_config,
            )
            vq_config["name"] = "finite_scalar_quantizer"
            return quantizer
        else:
            raise NotImplemented("quantizer {} not implemented".format(name))

    def quantize_enc_outs(self, x):
        ret_dict = {}

        if self.normalized_quant_input:
            x = F.normalize(x, dim=-1)
        ret_dict["quant_in"] = x
        x, indices, commit_loss, sub_quants = self.quantizer(x)
        ret_dict["quant_out"] = x
        ret_dict["indices"] = indices
        ret_dict["quant_loss"] = commit_loss

        return x, ret_dict

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor = None,
        **kwargs,
    ):
        use_padmask = self.use_padmask
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        only_extract_tokens = kwargs.get("only_extract_tokens", False)

        n_frames = x.size(1)
        max_pos = n_frames

        if ilens is not None:
            if self.downsample_rate == 4:
                olens = (
                    1
                    + (ilens - self.conv1.kernel_size[0] + 2 * self.conv1.padding[0])
                    // self.conv1.stride[0]
                )
            else:
                olens = ilens
            olens = (
                1
                + (olens - self.conv2.kernel_size[0] + 2 * self.conv2.padding[0])
                // self.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        if use_padmask and olens is not None:
            padding_mask = (~make_pad_mask(olens)[:, None, :]).to(torch.bool).to(x.device)
        else:
            padding_mask = None

        device = x.device
        seq_length = x.shape[1]
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        for layer, block in enumerate(self.blocks):
            x = block(x, mask=padding_mask, position_ids=position_ids)
            if self.quantize_layer_idx is not None and self.quantizer is not None:
                if layer == self.quantize_layer_idx:
                    hint_once(
                        f"Quantization at layer {layer} wit {self.quantizer}",
                        "normalize_quant_enc_out",
                        rank=0,
                    )
                    x, ret_dict = self.quantize_enc_outs(x)
                    if only_extract_tokens:
                        return (x, ret_dict), olens

        x = self.ln_post(x)

        if ilens is None:
            return x
        else:
            return x, olens


import types
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.cuda.amp import autocast
from funasr.metrics.compute_acc import compute_accuracy, th_accuracy
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.train_utils.device_funcs import force_gatherable
from . import whisper_lib as whisper
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils.datadir_writer import DatadirWriter
import logging


@tables.register("model_classes", "SenseVoiceLExtractTokens")
class SenseVoiceLExtractTokens(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        encoder = kwargs.get("encoder")
        encoder_conf = kwargs.get("encoder_conf", {})
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(**encoder_conf)

        if encoder_conf.get("freeze", False):
            freeze_exclude_key = encoder_conf.get("freeze_exclude_key", None)
            for name, param in encoder.named_parameters():
                if not freeze_exclude_key in name:
                    logging.info(f"name: {name} is freeze")
                    param.requires_grad = False

        dims = kwargs.get("dims", {})
        dims = whisper.model.ModelDimensions(**dims)
        model = whisper.model.Whisper(dims=dims)

        # encoder
        del model.encoder
        model.encoder = encoder

        # decoder
        model.decoder.use_padmask = kwargs.get("use_padmask", True)
        from .decoder import sense_voice_decode_forward

        model.decoder.forward = types.MethodType(sense_voice_decode_forward, model.decoder)

        self.model = model

        self.encoder_output_size = self.model.dims.n_audio_state

        self.activation_checkpoint = kwargs.get("activation_checkpoint", False)
        self.ignore_id = kwargs.get("ignore_id", -1)
        self.vocab_size = kwargs.get("vocab_size", -1)
        self.length_normalized_loss = kwargs.get("length_normalized_loss", True)
        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )

        specaug = kwargs.get("specaug", None)
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**kwargs.get("specaug_conf", {}))
        self.specaug = specaug

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        target_mask = kwargs.get("target_mask", None)

        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        if self.activation_checkpoint:
            from torch.utils.checkpoint import checkpoint

            encoder_out, encoder_out_lens = checkpoint(
                self.encode, speech, speech_lengths, use_reentrant=False
            )
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths, target_mask=target_mask
        )
        loss = loss_att
        stats = {}
        stats["acc"] = acc_att
        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        """Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):

            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

        # Forward encoder
        encoder_out, encoder_out_lens = self.model.encoder(speech.permute(0, 2, 1), speech_lengths)

        return encoder_out, encoder_out_lens

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        **kwargs,
    ):
        target_mask = kwargs.get("target_mask", None)
        stats = {}

        # 1. Forward decoder
        decoder_out = self.model.decoder(
            x=ys_pad, xa=encoder_out, hlens=encoder_out_lens, ys_in_lens=ys_pad_lens
        )

        # 2. Compute attention loss
        mask = torch.ones_like(ys_pad) * (-1)
        ys_pad_mask = (ys_pad * target_mask + mask * (1 - target_mask)).to(torch.int64)
        ys_pad_mask[ys_pad_mask == 0] = -1
        loss_att = self.criterion_att(decoder_out[:, :-1, :], ys_pad_mask[:, 1:])

        with torch.no_grad():
            preds = torch.argmax(decoder_out, -1)
            acc_att = compute_accuracy(
                preds[:, :-1], ys_pad_mask[:, 1:], ignore_label=self.ignore_id
            )

        return loss_att, acc_att, None, None

    # def inference(
    #     self,
    #     data_in,
    #     data_lengths=None,
    #     key: list = None,
    #     tokenizer=None,
    #     frontend=None,
    #     **kwargs,
    # ):
    #     if kwargs.get("batch_size", 1) > 1:
    #         raise NotImplementedError("batch decoding is not implemented")
    #
    #     if frontend is None and not hasattr(self, "frontend"):
    #         frontend_class = tables.frontend_classes.get("WhisperFrontend")
    #         frontend = frontend_class(
    #             n_mels=self.model.dims.n_mels, do_pad_trim=kwargs.get("do_pad_trim", True)
    #         )
    #         self.frontend = frontend
    #     else:
    #         frontend = frontend if frontend is not None else self.frontend
    #
    #     meta_data = {}
    #     if (
    #         isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
    #     ):  # fbank
    #         speech, speech_lengths = data_in, data_lengths
    #         if len(speech.shape) < 3:
    #             speech = speech[None, :, :]
    #         if speech_lengths is None:
    #             speech_lengths = speech.shape[1]
    #     else:
    #         # extract fbank feats
    #         time1 = time.perf_counter()
    #         audio_sample_list = load_audio_text_image_video(
    #             data_in,
    #             fs=frontend.fs if hasattr(frontend, "fs") else 16000,
    #             audio_fs=kwargs.get("fs", 16000),
    #             data_type=kwargs.get("data_type", "sound"),
    #             tokenizer=tokenizer,
    #         )
    #         time2 = time.perf_counter()
    #         meta_data["load_data"] = f"{time2 - time1:0.3f}"
    #         speech, speech_lengths = extract_fbank(
    #             audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
    #         )
    #         time3 = time.perf_counter()
    #         meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
    #         frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
    #         lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
    #         meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000
    #
    #     speech = speech.to(device=kwargs["device"])[0, :, :]
    #     speech_lengths = speech_lengths.to(device=kwargs["device"])
    #
    #     DecodingOptions = kwargs.get("DecodingOptions", {"fp16": kwargs.get("fp16", True)})
    #     task = DecodingOptions.get("task", "ASR")
    #     if isinstance(task, str):
    #         task = [task]
    #     task = "".join([f"<|{x}|>" for x in task])
    #     initial_prompt = kwargs.get("initial_prompt", f"<|startoftranscript|>{task}")
    #     DecodingOptions["initial_prompt"] = initial_prompt
    #
    #     language = DecodingOptions.get("language", None)
    #     language = None if language == "auto" else language
    #     DecodingOptions["language"] = language
    #
    #     DecodingOptions["vocab_path"] = kwargs["tokenizer_conf"].get("vocab_path", None)
    #
    #     if "without_timestamps" not in DecodingOptions:
    #         DecodingOptions["without_timestamps"] = True
    #
    #     options = whisper.DecodingOptions(**DecodingOptions)
    #
    #     result = whisper.decode(self.model, speech, options)
    #     text = f"{result.text}"
    #     results = []
    #     result_i = {"key": key[0], "text": text}
    #
    #     results.append(result_i)
    #
    #     ibest_writer = None
    #     if kwargs.get("output_dir") is not None:
    #         if not hasattr(self, "writer"):
    #             self.writer = DatadirWriter(kwargs.get("output_dir"))
    #         ibest_writer = self.writer[f"1best_recog"]
    #     if ibest_writer is not None:
    #         ibest_writer["text"][key[0]] = text
    #
    #     return results, meta_data

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        if frontend is None and not hasattr(self, "frontend"):
            frontend_class = tables.frontend_classes.get("WhisperFrontend")
            frontend = frontend_class(
                n_mels=self.model.dims.n_mels, do_pad_trim=kwargs.get("do_pad_trim", True)
            )
            self.frontend = frontend
        else:
            frontend = frontend if frontend is not None else self.frontend

        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs if hasattr(frontend, "fs") else 16000,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            if data_lengths is None:
                data_lengths = [x.shape[0] for x in audio_sample_list]
            speech, speech_lengths = extract_fbank(
                audio_sample_list,
                data_type=kwargs.get("data_type", "sound"),
                frontend=frontend,
                data_len=data_lengths,
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
            lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        (outs, ret_dict), out_lens = self.model.encoder(
            speech, speech_lengths, only_extract_tokens=True
        )
        time4 = time.perf_counter()
        meta_data["extract_tokens"] = f"{time4 - time3:0.3f}"
        # print(f'extract_tokens: {meta_data["extract_tokens"]}')
        tokens = ret_dict["indices"]

        text = "extract_token"
        results = []
        result_i = {"key": key[0], "text": text}

        # results.append(result_i)

        ark_writer, len_writer = None, None
        if kwargs.get("output_dir") is not None:
            out_dir = kwargs.get("output_dir")
            os.makedirs(out_dir, exist_ok=True)
            if not hasattr(self, "writer"):
                out_path = os.path.join(out_dir, f"enc_token")
                self.writer = kaldiio.WriteHelper(f"ark,scp,f:{out_path}.ark,{out_path}.scp")
                self.len_writer = open(out_path + "_len.txt", "wt")
            ark_writer = self.writer
            len_writer = self.len_writer

        if ark_writer is not None:
            for k, v, l in zip(key, tokens.detach().cpu().numpy(), out_lens):
                ark_writer(k, v[:l])
                len_writer.write(f"{k}\t{l}\n")
            time5 = time.perf_counter()
            meta_data["write_tokens"] = f"{time5 - time4:0.3f}"

        return results, meta_data
