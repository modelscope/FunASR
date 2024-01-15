# import torch
# from torch import nn
# from torch import Tensor
# import logging
# import numpy as np
# from funasr.train_utils.device_funcs import to_device
# from funasr.models.transformer.utils.nets_utils import make_pad_mask
# from funasr.models.scama.utils import sequence_mask
# from typing import Optional, Tuple
#
# from funasr.register import tables
#
# class mae_loss(nn.Module):
#
#     def __init__(self, normalize_length=False):
#         super(mae_loss, self).__init__()
#         self.normalize_length = normalize_length
#         self.criterion = torch.nn.L1Loss(reduction='sum')
#
#     def forward(self, token_length, pre_token_length):
#         loss_token_normalizer = token_length.size(0)
#         if self.normalize_length:
#             loss_token_normalizer = token_length.sum().type(torch.float32)
#         loss = self.criterion(token_length, pre_token_length)
#         loss = loss / loss_token_normalizer
#         return loss
#
#
# def cif(hidden, alphas, threshold):
#     batch_size, len_time, hidden_size = hidden.size()
#
#     # loop varss
#     integrate = torch.zeros([batch_size], device=hidden.device)
#     frame = torch.zeros([batch_size, hidden_size], device=hidden.device)
#     # intermediate vars along time
#     list_fires = []
#     list_frames = []
#
#     for t in range(len_time):
#         alpha = alphas[:, t]
#         distribution_completion = torch.ones([batch_size], device=hidden.device) - integrate
#
#         integrate += alpha
#         list_fires.append(integrate)
#
#         fire_place = integrate >= threshold
#         integrate = torch.where(fire_place,
#                                 integrate - torch.ones([batch_size], device=hidden.device),
#                                 integrate)
#         cur = torch.where(fire_place,
#                           distribution_completion,
#                           alpha)
#         remainds = alpha - cur
#
#         frame += cur[:, None] * hidden[:, t, :]
#         list_frames.append(frame)
#         frame = torch.where(fire_place[:, None].repeat(1, hidden_size),
#                             remainds[:, None] * hidden[:, t, :],
#                             frame)
#
#     fires = torch.stack(list_fires, 1)
#     frames = torch.stack(list_frames, 1)
#     list_ls = []
#     len_labels = torch.round(alphas.sum(-1)).int()
#     max_label_len = len_labels.max()
#     for b in range(batch_size):
#         fire = fires[b, :]
#         l = torch.index_select(frames[b, :, :], 0, torch.nonzero(fire >= threshold).squeeze())
#         pad_l = torch.zeros([max_label_len - l.size(0), hidden_size], device=hidden.device)
#         list_ls.append(torch.cat([l, pad_l], 0))
#     return torch.stack(list_ls, 0), fires
#
#
# def cif_wo_hidden(alphas, threshold):
#     batch_size, len_time = alphas.size()
#
#     # loop varss
#     integrate = torch.zeros([batch_size], device=alphas.device)
#     # intermediate vars along time
#     list_fires = []
#
#     for t in range(len_time):
#         alpha = alphas[:, t]
#
#         integrate += alpha
#         list_fires.append(integrate)
#
#         fire_place = integrate >= threshold
#         integrate = torch.where(fire_place,
#                                 integrate - torch.ones([batch_size], device=alphas.device)*threshold,
#                                 integrate)
#
#     fires = torch.stack(list_fires, 1)
#     return fires
#
# @tables.register("predictor_classes", "BATPredictor")
# class BATPredictor(nn.Module):
#     def __init__(self, idim, l_order, r_order, threshold=1.0, dropout=0.1, smooth_factor=1.0, noise_threshold=0, return_accum=False):
#         super(BATPredictor, self).__init__()
#
#         self.pad = nn.ConstantPad1d((l_order, r_order), 0)
#         self.cif_conv1d = nn.Conv1d(idim, idim, l_order + r_order + 1, groups=idim)
#         self.cif_output = nn.Linear(idim, 1)
#         self.dropout = torch.nn.Dropout(p=dropout)
#         self.threshold = threshold
#         self.smooth_factor = smooth_factor
#         self.noise_threshold = noise_threshold
#         self.return_accum = return_accum
#
#     def cif(
#         self,
#         input: Tensor,
#         alpha: Tensor,
#         beta: float = 1.0,
#         return_accum: bool = False,
#     ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
#         B, S, C = input.size()
#         assert tuple(alpha.size()) == (B, S), f"{alpha.size()} != {(B, S)}"
#
#         dtype = alpha.dtype
#         alpha = alpha.float()
#
#         alpha_sum = alpha.sum(1)
#         feat_lengths = (alpha_sum / beta).floor().long()
#         T = feat_lengths.max()
#
#         # aggregate and integrate
#         csum = alpha.cumsum(-1)
#         with torch.no_grad():
#             # indices used for scattering
#             right_idx = (csum / beta).floor().long().clip(max=T)
#             left_idx = right_idx.roll(1, dims=1)
#             left_idx[:, 0] = 0
#
#             # count # of fires from each source
#             fire_num = right_idx - left_idx
#             extra_weights = (fire_num - 1).clip(min=0)
#             # The extra entry in last dim is for
#             output = input.new_zeros((B, T + 1, C))
#             source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(input)
#             zero = alpha.new_zeros((1,))
#
#         # right scatter
#         fire_mask = fire_num > 0
#         right_weight = torch.where(
#             fire_mask,
#             csum - right_idx.type_as(alpha) * beta,
#             zero
#         ).type_as(input)
#         # assert right_weight.ge(0).all(), f"{right_weight} should be non-negative."
#         output.scatter_add_(
#             1,
#             right_idx.unsqueeze(-1).expand(-1, -1, C),
#             right_weight.unsqueeze(-1) * input
#         )
#
#         # left scatter
#         left_weight = (
#             alpha - right_weight - extra_weights.type_as(alpha) * beta
#         ).type_as(input)
#         output.scatter_add_(
#             1,
#             left_idx.unsqueeze(-1).expand(-1, -1, C),
#             left_weight.unsqueeze(-1) * input
#         )
#
#          # extra scatters
#         if extra_weights.ge(0).any():
#             extra_steps = extra_weights.max().item()
#             tgt_idx = left_idx
#             src_feats = input * beta
#             for _ in range(extra_steps):
#                 tgt_idx = (tgt_idx + 1).clip(max=T)
#                 # (B, S, 1)
#                 src_mask = (extra_weights > 0)
#                 output.scatter_add_(
#                     1,
#                     tgt_idx.unsqueeze(-1).expand(-1, -1, C),
#                     src_feats * src_mask.unsqueeze(2)
#                 )
#                 extra_weights -= 1
#
#         output = output[:, :T, :]
#
#         if return_accum:
#             return output, csum
#         else:
#             return output, alpha
#
#     def forward(self, hidden, target_label=None, mask=None, ignore_id=-1, mask_chunk_predictor=None, target_label_length=None):
#         h = hidden
#         context = h.transpose(1, 2)
#         queries = self.pad(context)
#         memory = self.cif_conv1d(queries)
#         output = memory + context
#         output = self.dropout(output)
#         output = output.transpose(1, 2)
#         output = torch.relu(output)
#         output = self.cif_output(output)
#         alphas = torch.sigmoid(output)
#         alphas = torch.nn.functional.relu(alphas*self.smooth_factor - self.noise_threshold)
#         if mask is not None:
#             alphas = alphas * mask.transpose(-1, -2).float()
#         if mask_chunk_predictor is not None:
#             alphas = alphas * mask_chunk_predictor
#         alphas = alphas.squeeze(-1)
#         if target_label_length is not None:
#             target_length = target_label_length
#         elif target_label is not None:
#             target_length = (target_label != ignore_id).float().sum(-1)
#             # logging.info("target_length: {}".format(target_length))
#         else:
#             target_length = None
#         token_num = alphas.sum(-1)
#         if target_length is not None:
#             # length_noise = torch.rand(alphas.size(0), device=alphas.device) - 0.5
#             # target_length = length_noise + target_length
#             alphas *= ((target_length + 1e-4) / token_num)[:, None].repeat(1, alphas.size(1))
#         acoustic_embeds, cif_peak = self.cif(hidden, alphas, self.threshold, self.return_accum)
#         return acoustic_embeds, token_num, alphas, cif_peak
