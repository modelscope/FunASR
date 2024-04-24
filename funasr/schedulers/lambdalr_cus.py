import torch
from torch.optim.lr_scheduler import _LRScheduler


class CustomLambdaLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * min(self.last_epoch / self.warmup_steps, 1) for base_lr in self.base_lrs
            ]
        else:
            return [base_lr for base_lr in self.base_lrs]


class CustomLambdaLR(_LRScheduler):
    def __init__(self, optimizer, train_config, last_epoch=-1, verbose=False):
        self.warmup_steps = train_config.warmup_steps
        self.total_steps = train_config.total_steps
        super(CustomLambdaLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        step = self._step_count
        if step < self.warmup_steps:
            lr_scale = step / self.warmup_steps
        else:
            lr_scale = max(
                0.0, 1 - (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            )
        return [base_lr * lr_scale for base_lr in self.base_lrs]
