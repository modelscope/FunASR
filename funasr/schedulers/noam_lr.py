"""Noam learning rate scheduler module."""

from typing import Union
import warnings

import torch
from torch.optim.lr_scheduler import _LRScheduler

from funasr.schedulers.abs_scheduler import AbsBatchStepScheduler


class NoamLR(_LRScheduler, AbsBatchStepScheduler):
    """The LR scheduler proposed by Noam

    Ref:
        "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf

    FIXME(kamo): PyTorch doesn't provide _LRScheduler as public class,
     thus the behaviour isn't guaranteed at forward PyTorch version.

    NOTE(kamo): The "model_size" in original implementation is derived from
     the model, but in this implementation, this parameter is a constant value.
     You need to change it if the model is changed.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: Union[int, float] = 320,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        self.model_size = model_size
        self.warmup_steps = warmup_steps

        lr = list(optimizer.param_groups)[0]["lr"]
        new_lr = self.lr_for_WarmupLR(lr)
        warnings.warn(
            f"NoamLR is deprecated. "
            f"Use WarmupLR(warmup_steps={warmup_steps}) with Optimizer(lr={new_lr})",
        )

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def lr_for_WarmupLR(self, lr: float) -> float:
        return lr / self.model_size**0.5 / self.warmup_steps**0.5

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(model_size={self.model_size}, "
            f"warmup_steps={self.warmup_steps})"
        )

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr * self.model_size**-0.5 * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]
