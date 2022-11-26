# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Tuple

import torch


class AbsESPnetModel(torch.nn.Module, ABC):
    """The common abstract class among each tasks

    "ESPnetModel" is referred to a class which inherits torch.nn.Module,
    and makes the dnn-models forward as its member field,
    a.k.a delegate pattern,
    and defines "loss", "stats", and "weight" for the task.

    If you intend to implement new task in ESPNet,
    the model must inherit this class.
    In other words, the "mediator" objects between
    our training system and the your task class are
    just only these three values, loss, stats, and weight.

    Example:
        >>> from funasr.tasks.abs_task import AbsTask
        >>> class YourESPnetModel(AbsESPnetModel):
        ...     def forward(self, input, input_lengths):
        ...         ...
        ...         return loss, stats, weight
        >>> class YourTask(AbsTask):
        ...     @classmethod
        ...     def build_model(cls, args: argparse.Namespace) -> YourESPnetModel:
    """

    def __init__(self):
        super().__init__()
        self.num_updates = 0

    @abstractmethod
    def forward(
        self, **batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def get_num_updates(self):
        return self.num_updates
