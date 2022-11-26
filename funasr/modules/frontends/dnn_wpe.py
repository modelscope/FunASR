from typing import Tuple

from pytorch_wpe import wpe_one_iteration
import torch
from torch_complex.tensor import ComplexTensor

from funasr.modules.frontends.mask_estimator import MaskEstimator
from funasr.modules.nets_utils import make_pad_mask


class DNN_WPE(torch.nn.Module):
    def __init__(
        self,
        wtype: str = "blstmp",
        widim: int = 257,
        wlayers: int = 3,
        wunits: int = 300,
        wprojs: int = 320,
        dropout_rate: float = 0.0,
        taps: int = 5,
        delay: int = 3,
        use_dnn_mask: bool = True,
        iterations: int = 1,
        normalization: bool = False,
    ):
        super().__init__()
        self.iterations = iterations
        self.taps = taps
        self.delay = delay

        self.normalization = normalization
        self.use_dnn_mask = use_dnn_mask

        self.inverse_power = True

        if self.use_dnn_mask:
            self.mask_est = MaskEstimator(
                wtype, widim, wlayers, wunits, wprojs, dropout_rate, nmask=1
            )

    def forward(
        self, data: ComplexTensor, ilens: torch.LongTensor
    ) -> Tuple[ComplexTensor, torch.LongTensor, ComplexTensor]:
        """The forward function

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq or Some dimension of the feature vector

        Args:
            data: (B, C, T, F)
            ilens: (B,)
        Returns:
            data: (B, C, T, F)
            ilens: (B,)
        """
        # (B, T, C, F) -> (B, F, C, T)
        enhanced = data = data.permute(0, 3, 2, 1)
        mask = None

        for i in range(self.iterations):
            # Calculate power: (..., C, T)
            power = enhanced.real**2 + enhanced.imag**2
            if i == 0 and self.use_dnn_mask:
                # mask: (B, F, C, T)
                (mask,), _ = self.mask_est(enhanced, ilens)
                if self.normalization:
                    # Normalize along T
                    mask = mask / mask.sum(dim=-1)[..., None]
                # (..., C, T) * (..., C, T) -> (..., C, T)
                power = power * mask

            # Averaging along the channel axis: (..., C, T) -> (..., T)
            power = power.mean(dim=-2)

            # enhanced: (..., C, T) -> (..., C, T)
            enhanced = wpe_one_iteration(
                data.contiguous(),
                power,
                taps=self.taps,
                delay=self.delay,
                inverse_power=self.inverse_power,
            )

            enhanced.masked_fill_(make_pad_mask(ilens, enhanced.real), 0)

        # (B, F, C, T) -> (B, T, C, F)
        enhanced = enhanced.permute(0, 3, 2, 1)
        if mask is not None:
            mask = mask.transpose(-1, -3)
        return enhanced, ilens, mask
