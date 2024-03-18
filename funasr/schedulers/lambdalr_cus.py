
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CustomLambdaLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * min(self.last_epoch / self.warmup_steps, 1)
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr for base_lr in self.base_lrs]