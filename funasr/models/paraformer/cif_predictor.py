#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import logging
import numpy as np

from funasr.register import tables
from funasr.train_utils.device_funcs import to_device
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from torch.cuda.amp import autocast


@tables.register("predictor_classes", "CifPredictor")
class CifPredictor(torch.nn.Module):
    def __init__(
        self,
        idim,
        l_order,
        r_order,
        threshold=1.0,
        dropout=0.1,
        smooth_factor=1.0,
        noise_threshold=0,
        tail_threshold=0.45,
    ):
        super().__init__()

        self.pad = torch.nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = torch.nn.Conv1d(idim, idim, l_order + r_order + 1, groups=idim)
        self.cif_output = torch.nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.tail_threshold = tail_threshold

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
                target_length = (target_label != ignore_id).float().sum(-1)
            else:
                target_length = None
            token_num = alphas.sum(-1)
            if target_length is not None:
                alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
            elif self.tail_threshold > 0.0:
                hidden, alphas, token_num = self.tail_process_fn(
                    hidden, alphas, token_num, mask=mask
                )

            acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)

            if target_length is None and self.tail_threshold > 0.0:
                token_num_int = torch.max(token_num).type(torch.int32).item()
                acoustic_embeds = acoustic_embeds[:, :token_num_int, :]

        return acoustic_embeds, token_num, alphas, cif_peak

    def tail_process_fn(self, hidden, alphas, token_num=None, mask=None):
        b, t, d = hidden.size()
        tail_threshold = self.tail_threshold
        if mask is not None:
            zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold = torch.tensor([tail_threshold], dtype=alphas.dtype).to(alphas.device)
            tail_threshold = torch.reshape(tail_threshold, (1, 1))
            alphas = torch.cat([alphas, tail_threshold], dim=1)
        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return hidden, alphas, token_num_floor

    def gen_frame_alignments(
        self, alphas: torch.Tensor = None, encoder_sequence_length: torch.Tensor = None
    ):
        batch_size, maximum_length = alphas.size()
        int_type = torch.int32

        is_training = self.training
        if is_training:
            token_num = torch.round(torch.sum(alphas, dim=1)).type(int_type)
        else:
            token_num = torch.floor(torch.sum(alphas, dim=1)).type(int_type)

        max_token_num = torch.max(token_num).item()

        alphas_cumsum = torch.cumsum(alphas, dim=1)
        alphas_cumsum = torch.floor(alphas_cumsum).type(int_type)
        alphas_cumsum = alphas_cumsum[:, None, :].repeat(1, max_token_num, 1)

        index = torch.ones([batch_size, max_token_num], dtype=int_type)
        index = torch.cumsum(index, dim=1)
        index = index[:, :, None].repeat(1, 1, maximum_length).to(alphas_cumsum.device)

        index_div = torch.floor(torch.true_divide(alphas_cumsum, index)).type(int_type)
        index_div_bool_zeros = index_div.eq(0)
        index_div_bool_zeros_count = torch.sum(index_div_bool_zeros, dim=-1) + 1
        index_div_bool_zeros_count = torch.clamp(
            index_div_bool_zeros_count, 0, encoder_sequence_length.max()
        )
        token_num_mask = (~make_pad_mask(token_num, maxlen=max_token_num)).to(token_num.device)
        index_div_bool_zeros_count *= token_num_mask

        index_div_bool_zeros_count_tile = index_div_bool_zeros_count[:, :, None].repeat(
            1, 1, maximum_length
        )
        ones = torch.ones_like(index_div_bool_zeros_count_tile)
        zeros = torch.zeros_like(index_div_bool_zeros_count_tile)
        ones = torch.cumsum(ones, dim=2)
        cond = index_div_bool_zeros_count_tile == ones
        index_div_bool_zeros_count_tile = torch.where(cond, zeros, ones)

        index_div_bool_zeros_count_tile_bool = index_div_bool_zeros_count_tile.type(torch.bool)
        index_div_bool_zeros_count_tile = 1 - index_div_bool_zeros_count_tile_bool.type(int_type)
        index_div_bool_zeros_count_tile_out = torch.sum(index_div_bool_zeros_count_tile, dim=1)
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out.type(int_type)
        predictor_mask = (
            (~make_pad_mask(encoder_sequence_length, maxlen=encoder_sequence_length.max()))
            .type(int_type)
            .to(encoder_sequence_length.device)
        )
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out * predictor_mask

        predictor_alignments = index_div_bool_zeros_count_tile_out
        predictor_alignments_length = predictor_alignments.sum(-1).type(
            encoder_sequence_length.dtype
        )
        return predictor_alignments.detach(), predictor_alignments_length.detach()


@tables.register("predictor_classes", "CifPredictorV2")
class CifPredictorV2(torch.nn.Module):
    def __init__(
        self,
        idim,
        l_order,
        r_order,
        threshold=1.0,
        dropout=0.1,
        smooth_factor=1.0,
        noise_threshold=0,
        tail_threshold=0.0,
        tf2torch_tensor_name_prefix_torch="predictor",
        tf2torch_tensor_name_prefix_tf="seq2seq/cif",
        tail_mask=True,
    ):
        super().__init__()

        self.pad = torch.nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = torch.nn.Conv1d(idim, idim, l_order + r_order + 1)
        self.cif_output = torch.nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.tail_threshold = tail_threshold
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf
        self.tail_mask = tail_mask

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
            output = torch.relu(self.cif_conv1d(queries))
            output = output.transpose(1, 2)

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
                target_length = target_label_length.squeeze(-1)
            elif target_label is not None:
                target_length = (target_label != ignore_id).float().sum(-1)
            else:
                target_length = None
            token_num = alphas.sum(-1)
            if target_length is not None:
                alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
            elif self.tail_threshold > 0.0:
                if self.tail_mask:
                    hidden, alphas, token_num = self.tail_process_fn(
                        hidden, alphas, token_num, mask=mask
                    )
                else:
                    hidden, alphas, token_num = self.tail_process_fn(
                        hidden, alphas, token_num, mask=None
                    )

            acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)
            if target_length is None and self.tail_threshold > 0.0:
                token_num_int = torch.max(token_num).type(torch.int32).item()
                acoustic_embeds = acoustic_embeds[:, :token_num_int, :]

        return acoustic_embeds, token_num, alphas, cif_peak

    def forward_chunk(self, hidden, cache=None, **kwargs):
        is_final = kwargs.get("is_final", False)
        batch_size, len_time, hidden_size = hidden.shape
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))
        output = output.transpose(1, 2)
        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)

        alphas = alphas.squeeze(-1)

        token_length = []
        list_fires = []
        list_frames = []
        cache_alphas = []
        cache_hiddens = []

        if cache is not None and "chunk_size" in cache:
            alphas[:, : cache["chunk_size"][0]] = 0.0
            if not is_final:
                alphas[:, sum(cache["chunk_size"][:2]) :] = 0.0
        if cache is not None and "cif_alphas" in cache and "cif_hidden" in cache:
            cache["cif_hidden"] = to_device(cache["cif_hidden"], device=hidden.device)
            cache["cif_alphas"] = to_device(cache["cif_alphas"], device=alphas.device)
            hidden = torch.cat((cache["cif_hidden"], hidden), dim=1)
            alphas = torch.cat((cache["cif_alphas"], alphas), dim=1)
        if cache is not None and is_final:
            tail_hidden = torch.zeros((batch_size, 1, hidden_size), device=hidden.device)
            tail_alphas = torch.tensor([[self.tail_threshold]], device=alphas.device)
            tail_alphas = torch.tile(tail_alphas, (batch_size, 1))
            hidden = torch.cat((hidden, tail_hidden), dim=1)
            alphas = torch.cat((alphas, tail_alphas), dim=1)

        len_time = alphas.shape[1]
        for b in range(batch_size):
            integrate = 0.0
            frames = torch.zeros((hidden_size), device=hidden.device)
            list_frame = []
            list_fire = []
            for t in range(len_time):
                alpha = alphas[b][t]
                if alpha + integrate < self.threshold:
                    integrate += alpha
                    list_fire.append(integrate)
                    frames += alpha * hidden[b][t]
                else:
                    frames += (self.threshold - integrate) * hidden[b][t]
                    list_frame.append(frames)
                    integrate += alpha
                    list_fire.append(integrate)
                    integrate -= self.threshold
                    frames = integrate * hidden[b][t]

            cache_alphas.append(integrate)
            if integrate > 0.0:
                cache_hiddens.append(frames / integrate)
            else:
                cache_hiddens.append(frames)

            token_length.append(torch.tensor(len(list_frame), device=alphas.device))
            list_fires.append(list_fire)
            list_frames.append(list_frame)

        cache["cif_alphas"] = torch.stack(cache_alphas, axis=0)
        cache["cif_alphas"] = torch.unsqueeze(cache["cif_alphas"], axis=0)
        cache["cif_hidden"] = torch.stack(cache_hiddens, axis=0)
        cache["cif_hidden"] = torch.unsqueeze(cache["cif_hidden"], axis=0)

        max_token_len = max(token_length)
        if max_token_len == 0:
            return hidden, torch.stack(token_length, 0), None, None
        list_ls = []
        for b in range(batch_size):
            pad_frames = torch.zeros(
                (max_token_len - token_length[b], hidden_size), device=alphas.device
            )
            if token_length[b] == 0:
                list_ls.append(pad_frames)
            else:
                list_frames[b] = torch.stack(list_frames[b])
                list_ls.append(torch.cat((list_frames[b], pad_frames), dim=0))

        cache["cif_alphas"] = torch.stack(cache_alphas, axis=0)
        cache["cif_alphas"] = torch.unsqueeze(cache["cif_alphas"], axis=0)
        cache["cif_hidden"] = torch.stack(cache_hiddens, axis=0)
        cache["cif_hidden"] = torch.unsqueeze(cache["cif_hidden"], axis=0)
        return torch.stack(list_ls, 0), torch.stack(token_length, 0), None, None

    def tail_process_fn(self, hidden, alphas, token_num=None, mask=None):
        b, t, d = hidden.size()
        tail_threshold = self.tail_threshold
        if mask is not None:
            zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold = torch.tensor([tail_threshold], dtype=alphas.dtype).to(alphas.device)
            tail_threshold = torch.reshape(tail_threshold, (1, 1))
            if b > 1:
                alphas = torch.cat([alphas, tail_threshold.repeat(b, 1)], dim=1)
            else:
                alphas = torch.cat([alphas, tail_threshold], dim=1)
        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return hidden, alphas, token_num_floor

    def gen_frame_alignments(
        self, alphas: torch.Tensor = None, encoder_sequence_length: torch.Tensor = None
    ):
        batch_size, maximum_length = alphas.size()
        int_type = torch.int32

        is_training = self.training
        if is_training:
            token_num = torch.round(torch.sum(alphas, dim=1)).type(int_type)
        else:
            token_num = torch.floor(torch.sum(alphas, dim=1)).type(int_type)

        max_token_num = torch.max(token_num).item()

        alphas_cumsum = torch.cumsum(alphas, dim=1)
        alphas_cumsum = torch.floor(alphas_cumsum).type(int_type)
        alphas_cumsum = alphas_cumsum[:, None, :].repeat(1, max_token_num, 1)

        index = torch.ones([batch_size, max_token_num], dtype=int_type)
        index = torch.cumsum(index, dim=1)
        index = index[:, :, None].repeat(1, 1, maximum_length).to(alphas_cumsum.device)

        index_div = torch.floor(torch.true_divide(alphas_cumsum, index)).type(int_type)
        index_div_bool_zeros = index_div.eq(0)
        index_div_bool_zeros_count = torch.sum(index_div_bool_zeros, dim=-1) + 1
        index_div_bool_zeros_count = torch.clamp(
            index_div_bool_zeros_count, 0, encoder_sequence_length.max()
        )
        token_num_mask = (~make_pad_mask(token_num, maxlen=max_token_num)).to(token_num.device)
        index_div_bool_zeros_count *= token_num_mask

        index_div_bool_zeros_count_tile = index_div_bool_zeros_count[:, :, None].repeat(
            1, 1, maximum_length
        )
        ones = torch.ones_like(index_div_bool_zeros_count_tile)
        zeros = torch.zeros_like(index_div_bool_zeros_count_tile)
        ones = torch.cumsum(ones, dim=2)
        cond = index_div_bool_zeros_count_tile == ones
        index_div_bool_zeros_count_tile = torch.where(cond, zeros, ones)

        index_div_bool_zeros_count_tile_bool = index_div_bool_zeros_count_tile.type(torch.bool)
        index_div_bool_zeros_count_tile = 1 - index_div_bool_zeros_count_tile_bool.type(int_type)
        index_div_bool_zeros_count_tile_out = torch.sum(index_div_bool_zeros_count_tile, dim=1)
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out.type(int_type)
        predictor_mask = (
            (~make_pad_mask(encoder_sequence_length, maxlen=encoder_sequence_length.max()))
            .type(int_type)
            .to(encoder_sequence_length.device)
        )
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out * predictor_mask

        predictor_alignments = index_div_bool_zeros_count_tile_out
        predictor_alignments_length = predictor_alignments.sum(-1).type(
            encoder_sequence_length.dtype
        )
        return predictor_alignments.detach(), predictor_alignments_length.detach()


@tables.register("predictor_classes", "CifPredictorV2Export")
class CifPredictorV2Export(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()

        self.pad = model.pad
        self.cif_conv1d = model.cif_conv1d
        self.cif_output = model.cif_output
        self.threshold = model.threshold
        self.smooth_factor = model.smooth_factor
        self.noise_threshold = model.noise_threshold
        self.tail_threshold = model.tail_threshold

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ):
        alphas, token_num = self.forward_cnn(hidden, mask)
        mask = mask.transpose(-1, -2).float()
        mask = mask.squeeze(-1)
        hidden, alphas, token_num = self.tail_process_fn(hidden, alphas, mask=mask)
        acoustic_embeds, cif_peak = cif_export(hidden, alphas, self.threshold)

        return acoustic_embeds, token_num, alphas, cif_peak

    def forward_cnn(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))
        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
        mask = mask.transpose(-1, -2).float()
        alphas = alphas * mask
        alphas = alphas.squeeze(-1)
        token_num = alphas.sum(-1)

        return alphas, token_num

    def tail_process_fn(self, hidden, alphas, token_num=None, mask=None):
        b, t, d = hidden.size()
        tail_threshold = self.tail_threshold

        zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
        ones_t = torch.ones_like(zeros_t)

        mask_1 = torch.cat([mask, zeros_t], dim=1)
        mask_2 = torch.cat([ones_t, mask], dim=1)
        mask = mask_2 - mask_1
        tail_threshold = mask * tail_threshold
        alphas = torch.cat([alphas, zeros_t], dim=1)
        alphas = torch.add(alphas, tail_threshold)

        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return hidden, alphas, token_num_floor


@torch.jit.script
def cif_export(hidden, alphas, threshold: float):
    batch_size, len_time, hidden_size = hidden.size()
    threshold = torch.tensor([threshold], dtype=alphas.dtype).to(alphas.device)

    # loop varss
    integrate = torch.zeros([batch_size], dtype=alphas.dtype, device=hidden.device)
    frame = torch.zeros([batch_size, hidden_size], dtype=hidden.dtype, device=hidden.device)
    # intermediate vars along time
    list_fires = []
    list_frames = []

    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = (
            torch.ones([batch_size], dtype=alphas.dtype, device=hidden.device) - integrate
        )

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place,
            integrate - torch.ones([batch_size], dtype=alphas.dtype, device=hidden.device),
            integrate,
        )
        cur = torch.where(fire_place, distribution_completion, alpha)
        remainds = alpha - cur

        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(
            fire_place[:, None].repeat(1, hidden_size), remainds[:, None] * hidden[:, t, :], frame
        )

    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)

    fire_idxs = fires >= threshold
    frame_fires = torch.zeros_like(hidden)
    max_label_len = frames[0, fire_idxs[0]].size(0)
    for b in range(batch_size):
        frame_fire = frames[b, fire_idxs[b]]
        frame_len = frame_fire.size(0)
        frame_fires[b, :frame_len, :] = frame_fire

        if frame_len >= max_label_len:
            max_label_len = frame_len
    frame_fires = frame_fires[:, :max_label_len, :]
    return frame_fires, fires


class mae_loss(torch.nn.Module):

    def __init__(self, normalize_length=False):
        super(mae_loss, self).__init__()
        self.normalize_length = normalize_length
        self.criterion = torch.nn.L1Loss(reduction="sum")

    def forward(self, token_length, pre_token_length):
        loss_token_normalizer = token_length.size(0)
        if self.normalize_length:
            loss_token_normalizer = token_length.sum().type(torch.float32)
        loss = self.criterion(token_length, pre_token_length)
        loss = loss / loss_token_normalizer
        return loss


def cif(hidden, alphas, threshold):
    batch_size, len_time, hidden_size = hidden.size()

    # loop varss
    integrate = torch.zeros([batch_size], device=hidden.device)
    frame = torch.zeros([batch_size, hidden_size], device=hidden.device)
    # intermediate vars along time
    list_fires = []
    list_frames = []

    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = torch.ones([batch_size], device=hidden.device) - integrate

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place, integrate - torch.ones([batch_size], device=hidden.device), integrate
        )
        cur = torch.where(fire_place, distribution_completion, alpha)
        remainds = alpha - cur

        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(
            fire_place[:, None].repeat(1, hidden_size), remainds[:, None] * hidden[:, t, :], frame
        )

    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)
    list_ls = []
    len_labels = torch.round(alphas.sum(-1)).int()
    max_label_len = len_labels.max()
    for b in range(batch_size):
        fire = fires[b, :]
        l = torch.index_select(frames[b, :, :], 0, torch.nonzero(fire >= threshold).squeeze())
        pad_l = torch.zeros([max_label_len - l.size(0), hidden_size], device=hidden.device)
        list_ls.append(torch.cat([l, pad_l], 0))
    return torch.stack(list_ls, 0), fires


def cif_wo_hidden(alphas, threshold):
    batch_size, len_time = alphas.size()

    # loop varss
    integrate = torch.zeros([batch_size], device=alphas.device)
    # intermediate vars along time
    list_fires = []

    for t in range(len_time):
        alpha = alphas[:, t]

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place,
            integrate - torch.ones([batch_size], device=alphas.device) * threshold,
            integrate,
        )

    fires = torch.stack(list_fires, 1)
    return fires
