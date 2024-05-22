"""Torch utility module."""

import torch

if torch.distributed.is_available():
    from torch.distributed import ReduceOp


def recursive_sum(obj, weight: torch.Tensor, distributed: bool = False):
    assert weight.dim() == 1, weight.size()
    if isinstance(obj, (tuple, list)):
        return type(obj)(recursive_sum(v, weight, distributed) for v in obj)
    elif isinstance(obj, dict):
        return {k: recursive_sum(v, weight, distributed) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        assert obj.size() == weight.size(), (obj.size(), weight.size())
        obj = (obj * weight.type(obj.dtype)).sum()
        if distributed:
            torch.distributed.all_reduce(obj, op=ReduceOp.SUM)
        return obj
    elif obj is None:
        return None
    else:
        raise ValueError(type(obj))


def recursive_divide(a, b: torch.Tensor):
    if isinstance(a, (tuple, list)):
        return type(a)(recursive_divide(v, b) for v in a)
    elif isinstance(a, dict):
        return {k: recursive_divide(v, b) for k, v in a.items()}
    elif isinstance(a, torch.Tensor):
        assert a.size() == b.size(), (a.size(), b.size())
        return a / b.type(a.dtype)
    elif a is None:
        return None
    else:
        raise ValueError(type(a))


def recursive_average(obj, weight: torch.Tensor, distributed: bool = False):
    obj = recursive_sum(obj, weight, distributed)
    weight = weight.sum()
    if distributed:
        torch.distributed.all_reduce(weight, op=ReduceOp.SUM)
    # Normalize weight to be sum-to-1
    obj = recursive_divide(obj, weight)
    return obj, weight
