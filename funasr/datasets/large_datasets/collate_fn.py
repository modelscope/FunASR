from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from funasr.models.transformer.utils.nets_utils import pad_list, pad_list_all_dim


class CommonCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
        max_sample_size=None,
    ):
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)
        self.max_sample_size = max_sample_size

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        return common_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )


def common_collate_fn(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor."""
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]
        tensor_list = [torch.from_numpy(a) for a in array_list]
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        if key not in not_sequence:
            lens = torch.tensor([d[key].shape[0] for d in data], dtype=torch.long)
            output[key + "_lengths"] = lens

    output = (uttids, output)
    return output


class DiarCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
        max_sample_size=None,
    ):
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)
        self.max_sample_size = max_sample_size

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        return diar_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )


def diar_collate_fn(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor."""
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]
        tensor_list = [torch.from_numpy(a) for a in array_list]
        tensor = pad_list_all_dim(tensor_list, pad_value)
        output[key] = tensor

        if key not in not_sequence:
            lens = torch.tensor([d[key].shape[0] for d in data], dtype=torch.long)
            output[key + "_lengths"] = lens

    output = (uttids, output)
    return output


def crop_to_max_size(feature, target_size):
    size = len(feature)
    diff = size - target_size
    if diff <= 0:
        return feature

    start = np.random.randint(0, diff + 1)
    end = size - diff + start
    return feature[start:end]


def clipping_collate_fn(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    max_sample_size=None,
    not_sequence: Collection[str] = (),
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    # mainly for pre-training
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
        array_list = [d[key] for d in data]
        tensor_list = [torch.from_numpy(a) for a in array_list]
        sizes = [len(s) for s in tensor_list]
        if max_sample_size is None:
            target_size = min(sizes)
        else:
            target_size = min(min(sizes), max_sample_size)
        tensor = tensor_list[0].new_zeros(len(tensor_list), target_size, tensor_list[0].shape[1])
        for i, (source, size) in enumerate(zip(tensor_list, sizes)):
            diff = size - target_size
            if diff == 0:
                tensor[i] = source
            else:
                tensor[i] = crop_to_max_size(source, target_size)
        output[key] = tensor

        if key not in not_sequence:
            lens = torch.tensor([source.shape[0] for source in tensor], dtype=torch.long)
            output[key + "_lengths"] = lens

    output = (uttids, output)
    return output
