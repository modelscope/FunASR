#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
# Copyright 2024 Kun Zou (chinazoukun@gmail.com). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import logging
import numpy as np

from funasr.register import tables
from funasr.train_utils.device_funcs import to_device
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from torch.cuda.amp import autocast


@tables.register("predictor_classes", "PifPredictor")
class PifPredictor(torch.nn.Module):
    """
    Author: Kun Zou, chinazoukun@gmail.com
    E-Paraformer: A Faster and Better Parallel Transformer for Non-autoregressive End-to-End Mandarin Speech Recognition
    https://www.isca-archive.org/interspeech_2024/zou24_interspeech.pdf
    """
    def __init__(
        self,
        idim,
        l_order,
        r_order,
        threshold=1.0,
        dropout=0.1,
        smooth_factor=1.0,
        noise_threshold=0,
        sigma=0.5,
        bias=0.0,
        sigma_heads=4,
    ):
        super().__init__()

        self.pad = torch.nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = torch.nn.Conv1d(idim, idim, l_order + r_order + 1, groups=idim)
        self.cif_output = torch.nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.sigma = torch.nn.Parameter(torch.tensor([sigma]*sigma_heads))
        self.bias = torch.nn.Parameter(torch.tensor([bias]*sigma_heads))
        self.sigma_heads = sigma_heads

    def forward(
        self,
        hidden,
        target_label=None,
        mask=None,
        ignore_id=-1,
        mask_chunk_predictor=None,
        target_label_length=None,
    ):

        with autocast(False):
            h = hidden
            context = h.transpose(1, 2)
            queries = self.pad(context)
            memory = self.cif_conv1d(queries)
            output = memory + context
            output = self.dropout(output)
            output = output.transpose(1, 2)
            output = torch.relu(output)
            output = self.cif_output(output)
            alphas = torch.sigmoid(output)
            alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
            if mask is not None:
                mask = mask.transpose(-1, -2).float()
                alphas = alphas * mask
            if mask_chunk_predictor is not None:
                alphas = alphas * mask_chunk_predictor
            alphas = alphas.squeeze(-1)
            mask = mask.squeeze(-1)
            if target_label_length is not None:
                target_length = target_label_length
            elif target_label is not None:
                target_mask = (target_label != ignore_id).float()
                target_length = target_mask.sum(-1)
            else:
                target_mask = None
                target_length = None
            token_num = alphas.sum(-1)
            if target_length is not None:
                alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
                max_token_num = torch.max(target_length)
            else:
                token_num_int = token_num.round()
                alphas *=(token_num_int / token_num)[:, None]
                max_token_num = torch.max(token_num_int)
            alignment = torch.cumsum(alphas, dim=-1)
            fire_positions = (torch.arange(max_token_num) + 0.5).type_as(alphas).unsqueeze(0)
            scores = - ((fire_positions[:, None, :, None] - alignment[:, None, None, :]) * self.sigma[None, :, None, None]) **2 + self.bias[None, :, None, None]
            scores = scores.masked_fill(~(mask[:, None, None, :].to(torch.bool)), float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            n_hidden = hidden.view(hidden.size(0), -1, self.sigma_heads, hidden.size(-1) // self.sigma_heads).transpose(1, 2)
            acoustic_embeds = torch.matmul(weights, n_hidden).transpose(1,2).contiguous().view(hidden.size(0), -1, hidden.size(-1))

            if target_mask is not None:
                acoustic_embeds *= target_mask[:, :, None]
            cif_peak = None
        return acoustic_embeds, token_num, alphas, cif_peak

