"""Warm up learning rate scheduler module."""

from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler

from funasr.schedulers.abs_scheduler import AbsBatchStepScheduler

class PartitionWarmupLR(_LRScheduler, AbsBatchStepScheduler):
    """The WarmupLR scheduler with support for fixed learning rates for certain parameter groups."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
        enable_warmup: list[bool] = None,  # 新增参数，表示每组是否启用 warmup
    ):
        self.warmup_steps = warmup_steps
        if enable_warmup is None:
            # 默认所有参数组都启用 warmup
            self.enable_warmup = [True] * len(optimizer.param_groups)
        else:
            self.enable_warmup = enable_warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        new_lrs = []
        for lr, is_warmup in zip(self.base_lrs, self.enable_warmup):
            if is_warmup:
                # 应用 warmup 调整
                new_lr = lr * self.warmup_steps**0.5 * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            else:
                # 保持原始的学习率
                new_lr = lr
            new_lrs.append(new_lr)
        return new_lrs