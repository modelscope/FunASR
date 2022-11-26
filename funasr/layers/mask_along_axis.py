import math
import torch
from typeguard import check_argument_types
from typing import Sequence
from typing import Union


def mask_along_axis(
    spec: torch.Tensor,
    spec_lengths: torch.Tensor,
    mask_width_range: Sequence[int] = (0, 30),
    dim: int = 1,
    num_mask: int = 2,
    replace_with_zero: bool = True,
):
    """Apply mask along the specified direction.

    Args:
        spec: (Batch, Length, Freq)
        spec_lengths: (Length): Not using lengths in this implementation
        mask_width_range: Select the width randomly between this range
    """

    org_size = spec.size()
    if spec.dim() == 4:
        # spec: (Batch, Channel, Length, Freq) -> (Batch * Channel, Length, Freq)
        spec = spec.view(-1, spec.size(2), spec.size(3))

    B = spec.shape[0]
    # D = Length or Freq
    D = spec.shape[dim]
    # mask_length: (B, num_mask, 1)
    mask_length = torch.randint(
        mask_width_range[0],
        mask_width_range[1],
        (B, num_mask),
        device=spec.device,
    ).unsqueeze(2)

    # mask_pos: (B, num_mask, 1)
    mask_pos = torch.randint(
        0, max(1, D - mask_length.max()), (B, num_mask), device=spec.device
    ).unsqueeze(2)

    # aran: (1, 1, D)
    aran = torch.arange(D, device=spec.device)[None, None, :]
    # mask: (Batch, num_mask, D)
    mask = (mask_pos <= aran) * (aran < (mask_pos + mask_length))
    # Multiply masks: (Batch, num_mask, D) -> (Batch, D)
    mask = mask.any(dim=1)
    if dim == 1:
        # mask: (Batch, Length, 1)
        mask = mask.unsqueeze(2)
    elif dim == 2:
        # mask: (Batch, 1, Freq)
        mask = mask.unsqueeze(1)

    if replace_with_zero:
        value = 0.0
    else:
        value = spec.mean()

    if spec.requires_grad:
        spec = spec.masked_fill(mask, value)
    else:
        spec = spec.masked_fill_(mask, value)
    spec = spec.view(*org_size)
    return spec, spec_lengths

def mask_along_axis_lfr(
    spec: torch.Tensor,
    spec_lengths: torch.Tensor,
    mask_width_range: Sequence[int] = (0, 30),
    dim: int = 1,
    num_mask: int = 2,
    replace_with_zero: bool = True,
    lfr_rate: int = 1,
):
    """Apply mask along the specified direction.

    Args:
        spec: (Batch, Length, Freq)
        spec_lengths: (Length): Not using lengths in this implementation
        mask_width_range: Select the width randomly between this range
        lfr_rate：low frame rate
    """

    org_size = spec.size()
    if spec.dim() == 4:
        # spec: (Batch, Channel, Length, Freq) -> (Batch * Channel, Length, Freq)
        spec = spec.view(-1, spec.size(2), spec.size(3))

    B = spec.shape[0]
    # D = Length or Freq
    D = spec.shape[dim] // lfr_rate
    # mask_length: (B, num_mask, 1)
    mask_length = torch.randint(
        mask_width_range[0],
        mask_width_range[1],
        (B, num_mask),
        device=spec.device,
    ).unsqueeze(2)
    if lfr_rate > 1:
        mask_length = mask_length.repeat(1, lfr_rate, 1)
    # mask_pos: (B, num_mask, 1)
    mask_pos = torch.randint(
        0, max(1, D - mask_length.max()), (B, num_mask), device=spec.device
    ).unsqueeze(2)
    if lfr_rate > 1:
        mask_pos_raw = mask_pos.clone()
        mask_pos = torch.zeros((B, 0, 1), device=spec.device, dtype=torch.int32)
        for i in range(lfr_rate):
            mask_pos_i = mask_pos_raw + D * i
            mask_pos = torch.cat((mask_pos, mask_pos_i), dim=1)
    # aran: (1, 1, D)
    D = spec.shape[dim]
    aran = torch.arange(D, device=spec.device)[None, None, :]
    # mask: (Batch, num_mask, D)
    mask = (mask_pos <= aran) * (aran < (mask_pos + mask_length))
    # Multiply masks: (Batch, num_mask, D) -> (Batch, D)
    mask = mask.any(dim=1)
    if dim == 1:
        # mask: (Batch, Length, 1)
        mask = mask.unsqueeze(2)
    elif dim == 2:
        # mask: (Batch, 1, Freq)
        mask = mask.unsqueeze(1)

    if replace_with_zero:
        value = 0.0
    else:
        value = spec.mean()

    if spec.requires_grad:
        spec = spec.masked_fill(mask, value)
    else:
        spec = spec.masked_fill_(mask, value)
    spec = spec.view(*org_size)
    return spec, spec_lengths


class MaskAlongAxis(torch.nn.Module):
    def __init__(
        self,
        mask_width_range: Union[int, Sequence[int]] = (0, 30),
        num_mask: int = 2,
        dim: Union[int, str] = "time",
        replace_with_zero: bool = True,
    ):
        assert check_argument_types()
        if isinstance(mask_width_range, int):
            mask_width_range = (0, mask_width_range)
        if len(mask_width_range) != 2:
            raise TypeError(
                f"mask_width_range must be a tuple of int and int values: "
                f"{mask_width_range}",
            )

        assert mask_width_range[1] > mask_width_range[0]
        if isinstance(dim, str):
            if dim == "time":
                dim = 1
            elif dim == "freq":
                dim = 2
            else:
                raise ValueError("dim must be int, 'time' or 'freq'")
        if dim == 1:
            self.mask_axis = "time"
        elif dim == 2:
            self.mask_axis = "freq"
        else:
            self.mask_axis = "unknown"

        super().__init__()
        self.mask_width_range = mask_width_range
        self.num_mask = num_mask
        self.dim = dim
        self.replace_with_zero = replace_with_zero

    def extra_repr(self):
        return (
            f"mask_width_range={self.mask_width_range}, "
            f"num_mask={self.num_mask}, axis={self.mask_axis}"
        )

    def forward(self, spec: torch.Tensor, spec_lengths: torch.Tensor = None):
        """Forward function.

        Args:
            spec: (Batch, Length, Freq)
        """

        return mask_along_axis(
            spec,
            spec_lengths,
            mask_width_range=self.mask_width_range,
            dim=self.dim,
            num_mask=self.num_mask,
            replace_with_zero=self.replace_with_zero,
        )


class MaskAlongAxisVariableMaxWidth(torch.nn.Module):
    """Mask input spec along a specified axis with variable maximum width.

    Formula:
        max_width = max_width_ratio * seq_len
    """

    def __init__(
        self,
        mask_width_ratio_range: Union[float, Sequence[float]] = (0.0, 0.05),
        num_mask: int = 2,
        dim: Union[int, str] = "time",
        replace_with_zero: bool = True,
    ):
        assert check_argument_types()
        if isinstance(mask_width_ratio_range, float):
            mask_width_ratio_range = (0.0, mask_width_ratio_range)
        if len(mask_width_ratio_range) != 2:
            raise TypeError(
                f"mask_width_ratio_range must be a tuple of float and float values: "
                f"{mask_width_ratio_range}",
            )

        assert mask_width_ratio_range[1] > mask_width_ratio_range[0]
        if isinstance(dim, str):
            if dim == "time":
                dim = 1
            elif dim == "freq":
                dim = 2
            else:
                raise ValueError("dim must be int, 'time' or 'freq'")
        if dim == 1:
            self.mask_axis = "time"
        elif dim == 2:
            self.mask_axis = "freq"
        else:
            self.mask_axis = "unknown"

        super().__init__()
        self.mask_width_ratio_range = mask_width_ratio_range
        self.num_mask = num_mask
        self.dim = dim
        self.replace_with_zero = replace_with_zero

    def extra_repr(self):
        return (
            f"mask_width_ratio_range={self.mask_width_ratio_range}, "
            f"num_mask={self.num_mask}, axis={self.mask_axis}"
        )

    def forward(self, spec: torch.Tensor, spec_lengths: torch.Tensor = None):
        """Forward function.

        Args:
            spec: (Batch, Length, Freq)
        """

        max_seq_len = spec.shape[self.dim]
        min_mask_width = math.floor(max_seq_len * self.mask_width_ratio_range[0])
        min_mask_width = max([0, min_mask_width])
        max_mask_width = math.floor(max_seq_len * self.mask_width_ratio_range[1])
        max_mask_width = min([max_seq_len, max_mask_width])

        if max_mask_width > min_mask_width:
            return mask_along_axis(
                spec,
                spec_lengths,
                mask_width_range=(min_mask_width, max_mask_width),
                dim=self.dim,
                num_mask=self.num_mask,
                replace_with_zero=self.replace_with_zero,
            )
        return spec, spec_lengths

class MaskAlongAxisLFR(torch.nn.Module):
    def __init__(
        self,
        mask_width_range: Union[int, Sequence[int]] = (0, 30),
        num_mask: int = 2,
        dim: Union[int, str] = "time",
        replace_with_zero: bool = True,
        lfr_rate: int = 1,
    ):
        assert check_argument_types()
        if isinstance(mask_width_range, int):
            mask_width_range = (0, mask_width_range)
        if len(mask_width_range) != 2:
            raise TypeError(
                f"mask_width_range must be a tuple of int and int values: "
                f"{mask_width_range}",
            )

        assert mask_width_range[1] > mask_width_range[0]
        if isinstance(dim, str):
            if dim == "time":
                dim = 1
                lfr_rate = 1
            elif dim == "freq":
                dim = 2
            else:
                raise ValueError("dim must be int, 'time' or 'freq'")
        if dim == 1:
            self.mask_axis = "time"
            lfr_rate = 1
        elif dim == 2:
            self.mask_axis = "freq"
        else:
            self.mask_axis = "unknown"

        super().__init__()
        self.mask_width_range = mask_width_range
        self.num_mask = num_mask
        self.dim = dim
        self.replace_with_zero = replace_with_zero
        self.lfr_rate = lfr_rate

    def extra_repr(self):
        return (
            f"mask_width_range={self.mask_width_range}, "
            f"num_mask={self.num_mask}, axis={self.mask_axis}"
        )

    def forward(self, spec: torch.Tensor, spec_lengths: torch.Tensor = None):
        """Forward function.

        Args:
            spec: (Batch, Length, Freq)
        """

        return mask_along_axis_lfr(
            spec,
            spec_lengths,
            mask_width_range=self.mask_width_range,
            dim=self.dim,
            num_mask=self.num_mask,
            replace_with_zero=self.replace_with_zero,
            lfr_rate=self.lfr_rate,
        )