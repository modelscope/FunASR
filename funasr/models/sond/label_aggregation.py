import torch
from typing import Optional
from typing import Tuple
from torch.nn import functional as F
from funasr.models.transformer.utils.nets_utils import make_pad_mask


class LabelAggregate(torch.nn.Module):
    def __init__(
        self,
        win_length: int = 512,
        hop_length: int = 128,
        center: bool = True,
    ):
        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center

    def extra_repr(self):
        return (
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """LabelAggregate forward function.

        Args:
            input: (Batch, Nsamples, Label_dim)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Label_dim)

        """
        bs = input.size(0)
        max_length = input.size(1)
        label_dim = input.size(2)

        # NOTE(jiatong):
        #   The default behaviour of label aggregation is compatible with
        #   torch.stft about framing and padding.

        # Step1: center padding
        if self.center:
            pad = self.win_length // 2
            max_length = max_length + 2 * pad
            input = torch.nn.functional.pad(input, (0, 0, pad, pad), "constant", 0)
            input[:, :pad, :] = input[:, pad : (2 * pad), :]
            input[:, (max_length - pad) : max_length, :] = input[
                :, (max_length - 2 * pad) : (max_length - pad), :
            ]
            nframe = (max_length - self.win_length) // self.hop_length + 1

        # Step2: framing
        output = input.as_strided(
            (bs, nframe, self.win_length, label_dim),
            (max_length * label_dim, self.hop_length * label_dim, label_dim, 1),
        )

        # Step3: aggregate label
        output = torch.gt(output.sum(dim=2, keepdim=False), self.win_length // 2)
        output = output.float()

        # Step4: process lengths
        if ilens is not None:
            if self.center:
                pad = self.win_length // 2
                ilens = ilens + 2 * pad

            olens = (ilens - self.win_length) // self.hop_length + 1
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output.to(input.dtype), olens


class LabelAggregateMaxPooling(torch.nn.Module):
    def __init__(
        self,
        hop_length: int = 8,
    ):
        super().__init__()

        self.hop_length = hop_length

    def extra_repr(self):
        return f"hop_length={self.hop_length}, "

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """LabelAggregate forward function.

        Args:
            input: (Batch, Nsamples, Label_dim)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Label_dim)

        """

        output = F.max_pool1d(input.transpose(1, 2), self.hop_length, self.hop_length).transpose(
            1, 2
        )
        olens = ilens // self.hop_length

        return output.to(input.dtype), olens
