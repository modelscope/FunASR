from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch

from funasr.modules.scorers.scorer_interface import BatchScorerInterface


class AbsPunctuation(torch.nn.Module, BatchScorerInterface, ABC):
    """The abstract class

    To share the loss calculation way among different models,
    We uses delegate pattern here:
    The instance of this class should be passed to "LanguageModel"

    >>> from funasr.punctuation.abs_model import AbsPunctuation
    >>> punc = AbsPunctuation()
    >>> model = ESPnetPunctuationModel(punc=punc)

    This "model" is one of mediator objects for "Task" class.

    """

    @abstractmethod
    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def with_vad(self) -> bool:
        raise NotImplementedError
