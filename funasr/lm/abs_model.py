from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch

from funasr.modules.scorers.scorer_interface import BatchScorerInterface


class AbsLM(torch.nn.Module, BatchScorerInterface, ABC):
    """The abstract LM class

    To share the loss calculation way among different models,
    We uses delegate pattern here:
    The instance of this class should be passed to "LanguageModel"

    >>> from funasr.lm.abs_model import AbsLM
    >>> lm = AbsLM()
    >>> model = LanguageESPnetModel(lm=lm)

    This "model" is one of mediator objects for "Task" class.

    """

    @abstractmethod
    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
