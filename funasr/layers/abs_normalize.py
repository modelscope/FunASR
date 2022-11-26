from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch


class AbsNormalize(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # return output, output_lengths
        raise NotImplementedError
