import os
import math

import torch
import torch.nn as nn










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

        self.attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(self.attn, value)  # (batch, head, time1, d_k)
        
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

        p = self.transpose_for_scores(self.linear_pos(pos_emb)) # (batch, head, time1, d_k)

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

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

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

        self.attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(self.attn, value)  # (batch, head, time1, d_k)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.linear_out(context_layer)  # (batch, time1, d_model)
        
