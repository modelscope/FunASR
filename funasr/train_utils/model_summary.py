import humanfriendly
import numpy as np
import torch


def get_human_readable_count(number: int) -> str:
    """Return human_readable_count

    Originated from:
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/core/memory.py

    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3 B'
        >>> get_human_readable_count(4e12)  # (four trillion)
        '4 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = [" ", "K", "M", "B", "T"]
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    return f"{number:.2f} {labels[index]}"


def to_bytes(dtype) -> int:
    # torch.float16 -> 16
    return int(str(dtype)[-2:]) // 8


def model_summary(model: torch.nn.Module) -> str:
    message = "Model structure:\n"
    message += str(model)
    tot_params = sum(p.numel() for p in model.parameters())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent_trainable = "{:.1f}".format(num_params * 100.0 / tot_params)
    tot_params = get_human_readable_count(tot_params)
    num_params = get_human_readable_count(num_params)
    message += "\n\nModel summary:\n"
    message += f"    Class Name: {model.__class__.__name__}\n"
    message += f"    Total Number of model parameters: {tot_params}\n"
    message += (
        f"    Number of trainable parameters: {num_params} ({percent_trainable}%)\n"
    )
    num_bytes = humanfriendly.format_size(
        sum(
            p.numel() * to_bytes(p.dtype) for p in model.parameters() if p.requires_grad
        )
    )
    message += f"    Size: {num_bytes}\n"
    dtype = next(iter(model.parameters())).dtype
    message += f"    Type: {dtype}"
    return message
