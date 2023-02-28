import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def padding(data, float_pad_value=0.0, int_pad_value=-1):
    assert isinstance(data, list)
    assert "key" in data[0]
    assert "speech" in data[0] or "text" in data[0]
    
    keys = [x["key"] for x in data]

    batch = {}
    data_names = data[0].keys()
    for data_name in data_names:
        if data_name == "key" or data_name =="sampling_rate":
            continue
        else:
            if data[0][data_name].dtype.kind == "i":
                pad_value = int_pad_value
                tensor_type = torch.int64
            else:
                pad_value = float_pad_value
                tensor_type = torch.float32

            tensor_list = [torch.tensor(np.copy(d[data_name]), dtype=tensor_type) for d in data]
            tensor_lengths = torch.tensor([len(d[data_name]) for d in data], dtype=torch.int32)
            tensor_pad = pad_sequence(tensor_list,
                                      batch_first=True,
                                      padding_value=pad_value)
            batch[data_name] = tensor_pad
            batch[data_name + "_lengths"] = tensor_lengths

    return keys, batch
