from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch

from funasr.modules.scorers.scorer_interface import ScorerInterface


class AbsDecoder(torch.nn.Module, ScorerInterface, ABC):
    @abstractmethod
    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
