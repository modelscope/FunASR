import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def standard_loss(ys, ts):
    losses = [F.binary_cross_entropy(torch.sigmoid(y), t) * len(y) for y, t in zip(ys, ts)]
    loss = torch.sum(torch.stack(losses))
    n_frames = (
        torch.from_numpy(np.array(np.sum([t.shape[0] for t in ts])))
        .to(torch.float32)
        .to(ys[0].device)
    )
    loss = loss / n_frames
    return loss


def fast_batch_pit_n_speaker_loss(ys, ts):
    with torch.no_grad():
        bs = len(ys)
        indices = []
        for b in range(bs):
            y = ys[b].transpose(0, 1)
            t = ts[b].transpose(0, 1)
            C, _ = t.shape
            y = y[:, None, :].repeat(1, C, 1)
            t = t[None, :, :].repeat(C, 1, 1)
            bce_loss = F.binary_cross_entropy(torch.sigmoid(y), t, reduction="none").mean(-1)
            C = bce_loss.cpu()
            indices.append(linear_sum_assignment(C))
    labels_perm = [t[:, idx[1]] for t, idx in zip(ts, indices)]

    return labels_perm


def cal_power_loss(logits, power_ts):
    losses = [
        F.cross_entropy(input=logit, target=power_t.to(torch.long)) * len(logit)
        for logit, power_t in zip(logits, power_ts)
    ]
    loss = torch.sum(torch.stack(losses))
    n_frames = (
        torch.from_numpy(np.array(np.sum([power_t.shape[0] for power_t in power_ts])))
        .to(torch.float32)
        .to(power_ts[0].device)
    )
    loss = loss / n_frames
    return loss
