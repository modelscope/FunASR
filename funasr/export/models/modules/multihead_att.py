import os
import math

import torch
import torch.nn as nn

class MultiHeadedAttentionSANM(nn.Module):
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
        q_h = q_h * self.d_k**(-0.5)
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

        self.attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(self.attn, value)  # (batch, head, time1, d_k)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.linear_out(context_layer)  # (batch, time1, d_model)

class MultiHeadedAttentionSANMDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fsmn_block = model.fsmn_block
        self.pad_fn = model.pad_fn
        self.kernel_size = model.kernel_size
        self.attn = None

    def forward(self, inputs, mask, cache=None):

        # b, t, d = inputs.size()
        # mask = torch.reshape(mask, (b, -1, 1))
        inputs = inputs * mask

        x = inputs.transpose(1, 2)
        if cache is None:
            x = self.pad_fn(x)
        else:
            x = torch.cat((cache[:, :, 1:], x), dim=2)
            cache = x
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)

        x = x + inputs
        x = x * mask
        return x, cache

class MultiHeadedAttentionCrossAtt(nn.Module):
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

        self.attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(self.attn, value)  # (batch, head, time1, d_k)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.linear_out(context_layer)  # (batch, time1, d_model)
