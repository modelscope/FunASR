import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
from itertools import combinations
from itertools import permutations


def generate_mapping_dict(max_speaker_num=6, max_olp_speaker_num=3):
    all_kinds = []
    all_kinds.append(0)
    for i in range(max_olp_speaker_num):
        selected_num = i + 1
        coms = np.array(list(combinations(np.arange(max_speaker_num), selected_num)))
        for com in coms:
            tmp = np.zeros(max_speaker_num)
            tmp[com] = 1
            item = int(raw_dec_trans(tmp.reshape(1, -1), max_speaker_num)[0])
            all_kinds.append(item)
    all_kinds_order = sorted(all_kinds)

    mapping_dict = {}
    mapping_dict["dec2label"] = {}
    mapping_dict["label2dec"] = {}
    for i in range(len(all_kinds_order)):
        dec = all_kinds_order[i]
        mapping_dict["dec2label"][dec] = i
        mapping_dict["label2dec"][i] = dec
    oov_id = len(all_kinds_order)
    mapping_dict["oov"] = oov_id
    return mapping_dict


def raw_dec_trans(x, max_speaker_num):
    num_list = []
    for i in range(max_speaker_num):
        num_list.append(x[:, i])
    base = 1
    T = x.shape[0]
    res = np.zeros((T))
    for num in num_list:
        res += num * base
        base = base * 2
    return res


def mapping_func(num, mapping_dict):
    if num in mapping_dict["dec2label"].keys():
        label = mapping_dict["dec2label"][num]
    else:
        label = mapping_dict["oov"]
    return label


def dec_trans(x, max_speaker_num, mapping_dict):
    num_list = []
    for i in range(max_speaker_num):
        num_list.append(x[:, i])
    base = 1
    T = x.shape[0]
    res = np.zeros((T))
    for num in num_list:
        res += num * base
        base = base * 2
    res = np.array([mapping_func(i, mapping_dict) for i in res])
    return res


def create_powerlabel(label, mapping_dict, max_speaker_num=6, max_olp_speaker_num=3):
    T, C = label.shape
    padding_label = np.zeros((T, max_speaker_num))
    padding_label[:, :C] = label
    out_label = dec_trans(padding_label, max_speaker_num, mapping_dict)
    out_label = torch.from_numpy(out_label)
    return out_label


def generate_perm_pse(label, n_speaker, mapping_dict, max_speaker_num, max_olp_speaker_num=3):
    perms = np.array(list(permutations(range(n_speaker)))).astype(np.float32)
    perms = torch.from_numpy(perms).to(label.device).to(torch.int64)
    perm_labels = [label[:, perm] for perm in perms]
    perm_pse_labels = [
        create_powerlabel(perm_label.cpu().numpy(), mapping_dict, max_speaker_num).to(
            perm_label.device, non_blocking=True
        )
        for perm_label in perm_labels
    ]
    return perm_labels, perm_pse_labels


def generate_min_pse(
    label, n_speaker, mapping_dict, max_speaker_num, pse_logit, max_olp_speaker_num=3
):
    perm_labels, perm_pse_labels = generate_perm_pse(
        label, n_speaker, mapping_dict, max_speaker_num, max_olp_speaker_num=max_olp_speaker_num
    )
    losses = [
        F.cross_entropy(input=pse_logit, target=perm_pse_label.to(torch.long)) * len(pse_logit)
        for perm_pse_label in perm_pse_labels
    ]
    loss = torch.stack(losses)
    min_index = torch.argmin(loss)
    selected_perm_label, selected_pse_label = perm_labels[min_index], perm_pse_labels[min_index]
    return selected_perm_label, selected_pse_label
