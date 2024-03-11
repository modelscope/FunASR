import torch
import torch.multiprocessing
import torch.nn
import torch.optim

from funasr.schedulers.noam_lr import NoamLR
from funasr.schedulers.tri_stage_scheduler import TriStageLR
from funasr.schedulers.warmup_lr import WarmupLR
from funasr.schedulers.lambdalr_cus import CustomLambdaLR

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
    custom_lambdalr=CustomLambdaLR,
)
