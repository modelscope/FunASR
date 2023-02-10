import io
from collections import OrderedDict
import numpy as np


def statistic_model_parameters(model, prefix=None):
    var_dict = model.state_dict()
    numel = 0
    for i, key in enumerate(sorted(list([x for x in var_dict.keys() if "num_batches_tracked" not in x]))):
        if prefix is None or key.startswith(prefix):
            numel += var_dict[key].numel()
    return numel


def int2vec(x, vec_dim=8, dtype=np.int):
    b = ('{:0' + str(vec_dim) + 'b}').format(x)
    # little-endian order: lower bit first
    return (np.array(list(b)[::-1]) == '1').astype(dtype)


def seq2arr(seq, vec_dim=8):
    return np.row_stack([int2vec(int(x), vec_dim) for x in seq])


def load_scp_as_dict(scp_path, value_type='str', kv_sep=" "):
    with io.open(scp_path, 'r', encoding='utf-8') as f:
        ret_dict = OrderedDict()
        for one_line in f.readlines():
            one_line = one_line.strip()
            pos = one_line.find(kv_sep)
            key, value = one_line[:pos], one_line[pos + 1:]
            if value_type == 'list':
                value = value.split(' ')
            ret_dict[key] = value
        return ret_dict


def load_scp_as_list(scp_path, value_type='str', kv_sep=" "):
    with io.open(scp_path, 'r', encoding='utf8') as f:
        ret_dict = []
        for one_line in f.readlines():
            one_line = one_line.strip()
            pos = one_line.find(kv_sep)
            key, value = one_line[:pos], one_line[pos + 1:]
            if value_type == 'list':
                value = value.split(' ')
            ret_dict.append((key, value))
        return ret_dict
