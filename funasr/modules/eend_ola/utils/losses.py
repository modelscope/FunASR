import numpy as np
import torch
import torch.nn.functional as F
from itertools import permutations
from torch import nn


def standard_loss(ys, ts, label_delay=0):
    losses = [F.binary_cross_entropy(torch.sigmoid(y), t) * len(y) for y, t in zip(ys, ts)]
    loss = torch.sum(torch.stack(losses))
    n_frames = torch.from_numpy(np.array(np.sum([t.shape[0] for t in ts]))).to(torch.float32).to(ys[0].device)
    loss = loss / n_frames
    return loss


def batch_pit_n_speaker_loss(ys, ts, n_speakers_list):
    max_n_speakers = ts[0].shape[1]
    olens = [y.shape[0] for y in ys]
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-1)
    ys_mask = [torch.ones(olen).to(ys.device) for olen in olens]
    ys_mask = torch.nn.utils.rnn.pad_sequence(ys_mask, batch_first=True, padding_value=0).unsqueeze(-1)

    losses = []
    for shift in range(max_n_speakers):
        ts_roll = [torch.roll(t, -shift, dims=1) for t in ts]
        ts_roll = nn.utils.rnn.pad_sequence(ts_roll, batch_first=True, padding_value=-1)
        loss = F.binary_cross_entropy(torch.sigmoid(ys), ts_roll, reduction='none')
        if ys_mask is not None:
            loss = loss * ys_mask
        loss = torch.sum(loss, dim=1)
        losses.append(loss)
    losses = torch.stack(losses, dim=2)

    perms = np.array(list(permutations(range(max_n_speakers)))).astype(np.float32)
    perms = torch.from_numpy(perms).to(losses.device)
    y_ind = torch.arange(max_n_speakers, dtype=torch.float32, device=losses.device)
    t_inds = torch.fmod(perms - y_ind, max_n_speakers).to(torch.long)

    losses_perm = []
    for t_ind in t_inds:
        losses_perm.append(
            torch.mean(losses[:, y_ind.to(torch.long), t_ind], dim=1))
    losses_perm = torch.stack(losses_perm, dim=1)

    def select_perm_indices(num, max_num):
        perms = list(permutations(range(max_num)))
        sub_perms = list(permutations(range(num)))
        return [
            [x[:num] for x in perms].index(perm)
            for perm in sub_perms]

    masks = torch.full_like(losses_perm, device=losses.device, fill_value=float('inf'))
    for i, t in enumerate(ts):
        n_speakers = n_speakers_list[i]
        indices = select_perm_indices(n_speakers, max_n_speakers)
        masks[i, indices] = 0
    losses_perm += masks

    min_loss = torch.sum(torch.min(losses_perm, dim=1)[0])
    n_frames = torch.from_numpy(np.array(np.sum([t.shape[0] for t in ts]))).to(losses.device)
    min_loss = min_loss / n_frames

    min_indices = torch.argmin(losses_perm, dim=1)
    labels_perm = [t[:, perms[idx].to(torch.long)] for t, idx in zip(ts, min_indices)]
    labels_perm = [t[:, :n_speakers] for t, n_speakers in zip(labels_perm, n_speakers_list)]

    return min_loss, labels_perm
