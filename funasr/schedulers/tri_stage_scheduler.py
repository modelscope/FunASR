# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, List

import torch
from torch.optim.lr_scheduler import _LRScheduler

from funasr.schedulers.abs_scheduler import AbsBatchStepScheduler


class TriStageLR(_LRScheduler, AbsBatchStepScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        phase_ratio: Optional[List[float]] = None,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.01,
    ):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.phase_ratio = phase_ratio
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale
        self.optimizer_lr = self.optimizer.defaults["lr"]

    def init_tri_stage_scheudler(self, max_update):
        self.max_update = max_update
        self.peak_lr = self.optimizer_lr
        self.init_lr = self.init_lr_scale * self.optimizer_lr
        self.final_lr = self.final_lr_scale * self.optimizer_lr

        assert self.max_update > 0
        assert sum(self.phase_ratio) == 1, "phase ratios must add up to 1"
        assert len(self.phase_ratio) == 3
        self.warmup_steps = int(self.max_update * self.phase_ratio[0])
        self.hold_steps = int(self.max_update * self.phase_ratio[1])
        self.decay_steps = int(self.max_update * self.phase_ratio[2])

        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        )
        self.decay_factor = -math.log(self.final_lr_scale) / self.decay_steps

        # initial learning rate
        self.lr = self.init_lr

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        self.set_optimizer_lr(self.lr)
        super().__init__(self.optimizer, self.last_epoch)

    def _decide_stage(self, update_step):
        """
        return stage, and the corresponding steps within the current stage
        """
        if update_step < self.warmup_steps:
            # warmup state
            return 0, update_step

        offset = self.warmup_steps

        if update_step < offset + self.hold_steps:
            # hold stage
            return 1, update_step - offset

        offset += self.hold_steps

        if update_step <= offset + self.decay_steps:
            # decay stage
            return 2, update_step - offset

        offset += self.decay_steps

        # still here ? constant lr stage
        return 3, update_step - offset

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        stage, steps_in_stage = self._decide_stage(num_updates)
        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")
        self.set_optimizer_lr(self.lr)

    def set_optimizer_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        step_num = self.last_epoch + 1
        self.step_update(step_num)
        return [self.lr]
