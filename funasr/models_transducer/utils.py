"""Utility functions for Transducer models."""

from typing import List, Tuple

import torch


class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling.

    Args:
        message: Error message to display.
        actual_size: The size that cannot pass the subsampling.
        limit: The size limit for subsampling.

    """

    def __init__(self, message: str, actual_size: int, limit: int) -> None:
        """Construct a TooShortUttError module."""
        super().__init__(message)

        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(sub_factor: int, size: int) -> Tuple[bool, int]:
    """Check if the input is too short for subsampling.

    Args:
        sub_factor: Subsampling factor for Conv2DSubsampling.
        size: Input size.

    Returns:
        : Whether an error should be sent.
        : Size limit for specified subsampling factor.

    """
    if sub_factor == 2 and size < 3:
        return True, 7
    elif sub_factor == 4 and size < 7:
        return True, 7
    elif sub_factor == 6 and size < 11:
        return True, 11

    return False, -1


def sub_factor_to_params(sub_factor: int, input_size: int) -> Tuple[int, int, int]:
    """Get conv2D second layer parameters for given subsampling factor.

    Args:
        sub_factor: Subsampling factor (1/X).
        input_size: Input size.

    Returns:
        : Kernel size for second convolution.
        : Stride for second convolution.
        : Conv2DSubsampling output size.

    """
    if sub_factor == 2:
        return 3, 1, (((input_size - 1) // 2 - 2))
    elif sub_factor == 4:
        return 3, 2, (((input_size - 1) // 2 - 1) // 2)
    elif sub_factor == 6:
        return 5, 3, (((input_size - 1) // 2 - 2) // 3)
    else:
        raise ValueError(
            "subsampling_factor parameter should be set to either 2, 4 or 6."
        )


def make_chunk_mask(
    size: int,
    chunk_size: int,
    left_chunk_size: int = 0,
    device: torch.device = None,
) -> torch.Tensor:
    """Create chunk mask for the subsequent steps (size, size).

    Reference: https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py

    Args:
        size: Size of the source mask.
        chunk_size: Number of frames in chunk.
        left_chunk_size: Size of the left context in chunks (0 means full context).
        device: Device for the mask tensor.

    Returns:
        mask: Chunk mask. (size, size)

    """
    mask = torch.zeros(size, size, device=device, dtype=torch.bool)

    for i in range(size):
        if left_chunk_size <= 0:
            start = 0
        else:
            start = max((i // chunk_size - left_chunk_size) * chunk_size, 0)

        end = min((i // chunk_size + 1) * chunk_size, size)
        mask[i, start:end] = True

    return ~mask


def make_source_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Create source mask for given lengths.

    Reference: https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py

    Args:
        lengths: Sequence lengths. (B,)

    Returns:
        : Mask for the sequence lengths. (B, max_len)

    """
    max_len = lengths.max()
    batch_size = lengths.size(0)

    expanded_lengths = torch.arange(max_len).expand(batch_size, max_len).to(lengths)

    return expanded_lengths >= lengths.unsqueeze(1)


def get_transducer_task_io(
    labels: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    ignore_id: int = -1,
    blank_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get Transducer loss I/O.

    Args:
        labels: Label ID sequences. (B, L)
        encoder_out_lens: Encoder output lengths. (B,)
        ignore_id: Padding symbol ID.
        blank_id: Blank symbol ID.

    Returns:
        decoder_in: Decoder inputs. (B, U)
        target: Target label ID sequences. (B, U)
        t_len: Time lengths. (B,)
        u_len: Label lengths. (B,)

    """

    def pad_list(labels: List[torch.Tensor], padding_value: int = 0):
        """Create padded batch of labels from a list of labels sequences.

        Args:
            labels: Labels sequences. [B x (?)]
            padding_value: Padding value.

        Returns:
            labels: Batch of padded labels sequences. (B,)

        """
        batch_size = len(labels)

        padded = (
            labels[0]
            .new(batch_size, max(x.size(0) for x in labels), *labels[0].size()[1:])
            .fill_(padding_value)
        )

        for i in range(batch_size):
            padded[i, : labels[i].size(0)] = labels[i]

        return padded

    device = labels.device

    labels_unpad = [y[y != ignore_id] for y in labels]
    blank = labels[0].new([blank_id])

    decoder_in = pad_list(
        [torch.cat([blank, label], dim=0) for label in labels_unpad], blank_id
    ).to(device)

    target = pad_list(labels_unpad, blank_id).type(torch.int32).to(device)

    encoder_out_lens = list(map(int, encoder_out_lens))
    t_len = torch.IntTensor(encoder_out_lens).to(device)

    u_len = torch.IntTensor([y.size(0) for y in labels_unpad]).to(device)

    return decoder_in, target, t_len, u_len

def pad_to_len(t: torch.Tensor, pad_len: int, dim: int):
    """Pad the tensor `t` at `dim` to the length `pad_len` with right padding zeros."""
    if t.size(dim) == pad_len:
        return t
    else:
        pad_size = list(t.shape)
        pad_size[dim] = pad_len - t.size(dim)
        return torch.cat(
            [t, torch.zeros(*pad_size, dtype=t.dtype, device=t.device)], dim=dim
        )
