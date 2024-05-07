import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.register import tables
import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from funasr.models.transformer.utils.mask import subsequent_mask


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


def sense_voice_decode_forward(
    self,
    x: torch.Tensor,
    xa: torch.Tensor,
    kv_cache: Optional[dict] = None,
    **kwargs,
):
    """Forward decoder.

    Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                    input token ids, int64 (batch, maxlen_out)
                    if input_layer == "embed"
                    input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
    Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                    if use_output_layer is True,
            olens: (batch, )
    """
    # import pdb;pdb.set_trace()
    use_padmask = self.use_padmask
    hlens = kwargs.get("hlens", None)

    ys_in_lens = kwargs.get("ys_in_lens", None)

    offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
    tgt, memory = x, xa
    tgt[tgt == -1] = 0
    tgt = self.token_embedding(tgt) + self.positional_embedding[offset : offset + tgt.size(1)]
    # tgt = self.dropout(tgt)

    x = tgt.to(memory.dtype)

    if use_padmask and hlens is not None:
        memory_mask = (~make_pad_mask(hlens)[:, None, :]).to(memory.device)
    else:
        memory_mask = None

    for layer, block in enumerate(self.blocks):
        x = block(
            x,
            memory,
            mask=self.mask,
            memory_mask=memory_mask,
            is_pad_mask=False,
            is_pad_memory_mask=True,
        )

    x = self.ln(x)
    x = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

    return x


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
                mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
                min_value = float(np.finfo(torch.tensor(0, dtype=qk.dtype).numpy().dtype).min)
                qk = qk.masked_fill(mask, min_value)

        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        if mask is not None and is_pad_mask:
            w = w.masked_fill(mask, 0.0)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


from omegaconf import OmegaConf


class ResidualAttentionBlockRWKV(nn.Module):
    def __init__(
        self, n_state: int, n_head: int, cross_attention: bool = False, layer_id=0, **kwargs
    ):
        super().__init__()

        rwkv_cfg = kwargs.get("rwkv_cfg", {})
        args = OmegaConf.create(rwkv_cfg)
        if args.get("version", "v4") == "v4":
            from funasr.models.sense_voice.rwkv_v4 import RWKVLayer
            from funasr.models.sense_voice.rwkv_v4 import RWKV_TimeMix as RWKV_Tmix
        elif args.get("version", "v5") == "v5":
            from funasr.models.sense_voice.rwkv_v5 import RWKVLayer
            from funasr.models.sense_voice.rwkv_v5 import RWKV_Tmix_x052 as RWKV_Tmix
        else:
            from funasr.models.sense_voice.rwkv_v6 import RWKVLayer
            from funasr.models.sense_voice.rwkv_v6 import RWKV_Tmix_x060 as RWKV_Tmix
        # self.att = RWKVLayer(args=args, layer_id=layer_id)
        self.att = RWKV_Tmix(args, layer_id=layer_id)

        if args.get("init_rwkv", True):
            print("init_rwkv")
            nn.init.orthogonal_(self.att.receptance.weight, gain=1)
            nn.init.orthogonal_(self.att.key.weight, gain=0.1)
            nn.init.orthogonal_(self.att.value.weight, gain=1)
            nn.init.orthogonal_(self.att.gate.weight, gain=0.1)
            nn.init.zeros_(self.att.output.weight)

        self.ln0 = None
        if layer_id == 0 and not args.get("ln0", True):
            self.ln0 = LayerNorm(args.n_embd)
            if args.get("init_rwkv", True):
                print("init_rwkv")
                layer_id = 0
                scale = ((1 + layer_id) / args.get("n_layer")) ** 0.7
                nn.init.constant_(self.ln0.weight, scale)

        self.layer_id = layer_id
        self.args = args

        self.ln1 = None
        if not args.get("ln1", True):
            self.ln1 = LayerNorm(args.n_embd)
            # init
            if args.get("init_rwkv", True):
                print("init_rwkv")
                scale = ((1 + layer_id) / args.get("n_layer")) ** 0.7
                nn.init.constant_(self.ln1.weight, scale)

        if args.get("datatype", "bf16") == "bf16":
            self.att.to(torch.bfloat16)
            # if self.ln1 is not None:
            #     self.ln1.to(torch.bfloat16)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        **kwargs,
    ):
        is_pad_mask = kwargs.get("is_pad_mask", False)
        is_pad_memory_mask = kwargs.get("is_pad_memory_mask", False)

        if self.layer_id == 0 and self.ln0 is not None:
            x = self.ln0(x)

        if self.args.get("datatype", "bf16") == "bf16":
            x = x.bfloat16()
        if self.ln1 is None:
            x = x + self.att(x, mask=mask, kv_cache=kv_cache, is_pad_mask=is_pad_mask)[0]
        else:
            x = x + self.att(self.ln1(x), mask=mask, kv_cache=kv_cache, is_pad_mask=is_pad_mask)[0]
        if self.args.get("datatype", "bf16") == "bf16":
            x = x.to(torch.float32)

        if self.cross_attn:
            x = (
                x
                + self.cross_attn(
                    self.cross_attn_ln(x), xa, kv_cache=kv_cache, is_pad_mask=is_pad_memory_mask
                )[0]
            )
        x = x + self.mlp(self.mlp_ln(x))

        return x


@tables.register("decoder_classes", "SenseVoiceDecoder")
class SenseVoiceDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, **kwargs):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlockRWKV(
                    n_state, n_head, cross_attention=True, layer_id=i, **kwargs
                )
                for i in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

        self.use_padmask = kwargs.get("use_padmask", True)

    def forward(
        self,
        x: torch.Tensor,
        xa: torch.Tensor,
        kv_cache: Optional[dict] = None,
        **kwargs,
    ):
        """Forward decoder.

        Args:
                hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
                hlens: (batch)
                ys_in_pad:
                        input token ids, int64 (batch, maxlen_out)
                        if input_layer == "embed"
                        input tensor (batch, maxlen_out, #mels) in the other cases
                ys_in_lens: (batch)
        Returns:
                (tuple): tuple containing:

                x: decoded token score before softmax (batch, maxlen_out, token)
                        if use_output_layer is True,
                olens: (batch, )
        """
        # import pdb;pdb.set_trace()
        use_padmask = self.use_padmask
        hlens = kwargs.get("hlens", None)

        ys_in_lens = kwargs.get("ys_in_lens", None)

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        tgt, memory = x, xa
        tgt[tgt == -1] = 0
        tgt = self.token_embedding(tgt) + self.positional_embedding[offset : offset + tgt.size(1)]
        # tgt = self.dropout(tgt)

        x = tgt.to(memory.dtype)

        if use_padmask and hlens is not None:
            memory_mask = (~make_pad_mask(hlens)[:, None, :]).to(memory.device)
        else:
            memory_mask = None

        for layer, block in enumerate(self.blocks):
            x = block(
                x,
                memory,
                mask=self.mask,
                memory_mask=memory_mask,
                is_pad_mask=False,
                is_pad_memory_mask=True,
            )

        x = self.ln(x)
        x = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return x

    def init_state(self, x):
        state = {}

        return state

    def final_score(self, state) -> float:
        """Score eos (optional).

        Args:
            state: Scorer state for prefix tokens

        Returns:
            float: final score

        """
        return 0.0

    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp = self.forward(ys.unsqueeze(0), x.unsqueeze(0), cache=state)
        return logp.squeeze(0)[-1, :], state


class MultiHeadedAttentionSANMDecoder(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_feat, dropout_rate, kernel_size, sanm_shfit=0):
        """Construct an MultiHeadedAttention object."""
        super().__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        # padding
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.kernel_size = kernel_size

    def forward(self, inputs, mask, cache=None, mask_shfit_chunk=None, **kwargs):
        """
        :param x: (#batch, time1, size).
        :param mask: Mask tensor (#batch, 1, time)
        :return:
        """
        # print("in fsmn, inputs", inputs.size())
        b, t, d = inputs.size()
        # logging.info(
        #     "mask: {}".format(mask.size()))
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            # logging.info("in fsmn, mask: {}, {}".format(mask.size(), mask[0:100:50, :, :]))
            if mask_shfit_chunk is not None:
                # logging.info("in fsmn, mask_fsmn: {}, {}".format(mask_shfit_chunk.size(), mask_shfit_chunk[0:100:50, :, :]))
                mask = mask * mask_shfit_chunk
            # logging.info("in fsmn, mask_after_fsmn: {}, {}".format(mask.size(), mask[0:100:50, :, :]))
            # print("in fsmn, mask", mask.size())
            # print("in fsmn, inputs", inputs.size())
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        b, d, t = x.size()
        if cache is None:
            # print("in fsmn, cache is None, x", x.size())

            x = self.pad_fn(x)
            if not self.training:
                cache = x
        else:
            # print("in fsmn, cache is not None, x", x.size())
            # x = torch.cat((x, cache), dim=2)[:, :, :-1]
            # if t < self.kernel_size:
            #     x = self.pad_fn(x)
            x = torch.cat((cache[:, :, 1:], x), dim=2)
            x = x[:, :, -(self.kernel_size + t - 1) :]
            # print("in fsmn, cache is not None, x_cat", x.size())
            cache = x
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        # print("in fsmn, fsmn_out", x.size())
        if x.size(1) != inputs.size(1):
            inputs = inputs[:, -1, :]

        x = x + inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x, cache


class ResidualAttentionBlockFSMN(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, **kwargs):
        super().__init__()

        self.attn = MultiHeadedAttentionSANMDecoder(
            n_state,
            kwargs.get("self_attention_dropout_rate"),
            kwargs.get("kernel_size", 20),
            kwargs.get("sanm_shfit", 10),
        )
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        **kwargs,
    ):
        cache = kwargs.get("cache", {})
        layer = kwargs.get("layer", 0)
        is_pad_mask = kwargs.get("is_pad_mask", False)
        is_pad_memory_mask = kwargs.get("is_pad_memory_mask", False)

        fsmn_cache = cache[layer]["fsmn_cache"] if cache is not None and len(cache) > 0 else None
        # if fsmn_cache is not None:
        #     x = x[:, -1:]
        att_res, fsmn_cache = self.attn(self.attn_ln(x), mask=None, cache=fsmn_cache)
        # if len(cache)>1:
        #     cache[layer]["fsmn_cache"] = fsmn_cache
        #     x = x[:, -1:]
        x = x + att_res
        if self.cross_attn:
            x = (
                x
                + self.cross_attn(
                    self.cross_attn_ln(x), xa, kv_cache=kv_cache, is_pad_mask=is_pad_memory_mask
                )[0]
            )
        x = x + self.mlp(self.mlp_ln(x))
        return x


@tables.register("decoder_classes", "SenseVoiceDecoderFSMN")
class SenseVoiceDecoderFSMN(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, **kwargs):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlockFSMN(
                    n_state, n_head, cross_attention=True, layer_id=i, **kwargs
                )
                for i in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

        self.use_padmask = kwargs.get("use_padmask", True)

    def forward(
        self,
        x: torch.Tensor,
        xa: torch.Tensor,
        kv_cache: Optional[dict] = None,
        **kwargs,
    ):
        """Forward decoder.

        Args:
                hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
                hlens: (batch)
                ys_in_pad:
                        input token ids, int64 (batch, maxlen_out)
                        if input_layer == "embed"
                        input tensor (batch, maxlen_out, #mels) in the other cases
                ys_in_lens: (batch)
        Returns:
                (tuple): tuple containing:

                x: decoded token score before softmax (batch, maxlen_out, token)
                        if use_output_layer is True,
                olens: (batch, )
        """
        # import pdb;pdb.set_trace()
        use_padmask = self.use_padmask
        hlens = kwargs.get("hlens", None)

        ys_in_lens = kwargs.get("ys_in_lens", None)

        tgt, memory = x, xa
        tgt[tgt == -1] = 0
        tgt = self.token_embedding(tgt) + self.positional_embedding[: tgt.size(1)]
        # tgt = self.dropout(tgt)

        x = tgt.to(memory.dtype)

        if use_padmask and hlens is not None:
            memory_mask = (~make_pad_mask(hlens)[:, None, :]).to(memory.device)
        else:
            memory_mask = None

        for layer, block in enumerate(self.blocks):
            x = block(
                x,
                memory,
                mask=self.mask,
                memory_mask=memory_mask,
                is_pad_mask=False,
                is_pad_memory_mask=True,
                cache=kwargs.get("cache", None),
                layer=layer,
            )

        x = self.ln(x)
        x = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return x

    def init_state(self, x):
        state = {}
        for layer, block in enumerate(self.blocks):
            state[layer] = {
                "fsmn_cache": None,
                "memory_key": None,
                "memory_value": None,
            }

        return state

    def final_score(self, state) -> float:
        """Score eos (optional).

        Args:
            state: Scorer state for prefix tokens

        Returns:
            float: final score

        """
        return 0.0

    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp = self.forward(ys.unsqueeze(0), x.unsqueeze(0), cache=None)
        logp = torch.log_softmax(logp, dim=-1)
        return logp.squeeze(0)[-1, :], state
