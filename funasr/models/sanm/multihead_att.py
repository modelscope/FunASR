import os
import math

import torch
import torch.nn as nn


class MultiHeadedAttentionSANMExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.d_k = model.d_k
        self.h = model.h
        self.linear_out = model.linear_out
        self.linear_q_k_v = model.linear_q_k_v
        self.fsmn_block = model.fsmn_block
        self.pad_fn = model.pad_fn

        self.attn = None
        self.all_head_size = self.h * self.d_k

    def forward(self, x, mask):
        mask_3d_btd, mask_4d_bhlt = mask
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask_3d_btd)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask_4d_bhlt)
        return att_outs + fsmn_memory

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward_qkv(self, x):
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = self.transpose_for_scores(q)
        k_h = self.transpose_for_scores(k)
        v_h = self.transpose_for_scores(v)
        return q_h, k_h, v_h, v

    def forward_fsmn(self, inputs, mask):
        # b, t, d = inputs.size()
        # mask = torch.reshape(mask, (b, -1, 1))
        inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x = x + inputs
        x = x * mask
        return x

    def forward_attention(self, value, scores, mask):
        scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(attn, value)  # (batch, head, time1, d_k)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.linear_out(context_layer)  # (batch, time1, d_model)


def preprocess_for_attn(x, mask, cache, pad_fn, kernel_size):
    x = x * mask
    x = x.transpose(1, 2)
    if cache is None:
        x = pad_fn(x)
    else:
        x = torch.cat((cache, x), dim=2)
        cache = x[:, :, -(kernel_size - 1) :]
    return x, cache


torch_version = tuple([int(i) for i in torch.__version__.split(".")[:2]])
if torch_version >= (1, 8):
    import torch.fx

    torch.fx.wrap("preprocess_for_attn")


class MultiHeadedAttentionSANMDecoderExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fsmn_block = model.fsmn_block
        self.pad_fn = model.pad_fn
        self.kernel_size = model.kernel_size
        self.attn = None

    def forward(self, inputs, mask, cache=None):
        x, cache = preprocess_for_attn(inputs, mask, cache, self.pad_fn, self.kernel_size)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)

        x = x + inputs
        x = x * mask
        return x, cache


class MultiHeadedAttentionCrossAttExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.d_k = model.d_k
        self.h = model.h
        self.linear_q = model.linear_q
        self.linear_k_v = model.linear_k_v
        self.linear_out = model.linear_out
        self.attn = None
        self.all_head_size = self.h * self.d_k

    def forward(self, x, memory, memory_mask):
        q, k, v = self.forward_qkv(x, memory)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, memory_mask)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward_qkv(self, x, memory):
        q = self.linear_q(x)

        k_v = self.linear_k_v(memory)
        k, v = torch.split(k_v, int(self.h * self.d_k), dim=-1)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(attn, value)  # (batch, head, time1, d_k)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.linear_out(context_layer)  # (batch, time1, d_model)


class OnnxMultiHeadedAttention(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.d_k = model.d_k
        self.h = model.h
        self.linear_q = model.linear_q
        self.linear_k = model.linear_k
        self.linear_v = model.linear_v
        self.linear_out = model.linear_out
        self.attn = None
        self.all_head_size = self.h * self.d_k

    def forward(self, query, key, value, mask):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward_qkv(self, query, key, value):
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(attn, value)  # (batch, head, time1, d_k)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.linear_out(context_layer)  # (batch, time1, d_model)


class OnnxRelPosMultiHeadedAttention(OnnxMultiHeadedAttention):
    def __init__(self, model):
        super().__init__(model)
        self.linear_pos = model.linear_pos
        self.pos_bias_u = model.pos_bias_u
        self.pos_bias_v = model.pos_bias_v

    def forward(self, query, key, value, pos_emb, mask):
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        p = self.transpose_for_scores(self.linear_pos(pos_emb))  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)

    def rel_shift(self, x):
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2
        return x

    def forward_attention(self, value, scores, mask):
        scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(attn, value)  # (batch, head, time1, d_k)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.linear_out(context_layer)  # (batch, time1, d_model)
