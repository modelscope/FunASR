import numpy as np
import torch

from funasr.datasets.large_datasets.collate_fn import crop_to_max_size


def clipping(data):
    assert isinstance(data, list)
    assert "key" in data[0]

    keys = [x["key"] for x in data]

    batch = {}
    data_names = data[0].keys()
    for data_name in data_names:
        if data_name == "key":
            continue
        else:
            if data[0][data_name].dtype.kind == "i":
                tensor_type = torch.int64
            else:
                tensor_type = torch.float32

            tensor_list = [torch.tensor(np.copy(d[data_name]), dtype=tensor_type) for d in data]
            tensor_lengths = torch.tensor([len(d[data_name]) for d in data], dtype=torch.int32)

            length_clip = min(tensor_lengths)
            tensor_clip = tensor_list[0].new_zeros(
                len(tensor_list), length_clip, tensor_list[0].shape[1]
            )
            for i, (tensor, length) in enumerate(zip(tensor_list, tensor_lengths)):
                diff = length - length_clip
                assert diff >= 0
                if diff == 0:
                    tensor_clip[i] = tensor
                else:
                    tensor_clip[i] = crop_to_max_size(tensor, length_clip)

            batch[data_name] = tensor_clip
            batch[data_name + "_lengths"] = torch.tensor(
                [tensor.shape[0] for tensor in tensor_clip], dtype=torch.long
            )

    return keys, batch
