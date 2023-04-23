import torch
import torch.multiprocessing
import torch.nn
import torch.optim

from funasr.schedulers.noam_lr import NoamLR
from funasr.schedulers.tri_stage_scheduler import TriStageLR
from funasr.schedulers.warmup_lr import WarmupLR


def build_scheduler(args, optimizers):
    scheduler_classes = dict(
        ReduceLROnPlateau=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lambdalr=torch.optim.lr_scheduler.LambdaLR,
        steplr=torch.optim.lr_scheduler.StepLR,
        multisteplr=torch.optim.lr_scheduler.MultiStepLR,
        exponentiallr=torch.optim.lr_scheduler.ExponentialLR,
        CosineAnnealingLR=torch.optim.lr_scheduler.CosineAnnealingLR,
        noamlr=NoamLR,
        warmuplr=WarmupLR,
        tri_stage=TriStageLR,
        cycliclr=torch.optim.lr_scheduler.CyclicLR,
        onecyclelr=torch.optim.lr_scheduler.OneCycleLR,
        CosineAnnealingWarmRestarts=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    )

    schedulers = []
    for i, optim in enumerate(optimizers, 1):
        suf = "" if i == 1 else str(i)
        name = getattr(args, f"scheduler{suf}")
        conf = getattr(args, f"scheduler{suf}_conf")
        if name is not None:
            cls_ = scheduler_classes.get(name)
            if cls_ is None:
                raise ValueError(
                    f"must be one of {list(scheduler_classes)}: {name}"
                )
            scheduler = cls_(optim, **conf)
        else:
            scheduler = None

        schedulers.append(scheduler)

    return schedulers