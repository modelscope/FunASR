from abc import ABC
from abc import abstractmethod

import torch.optim.lr_scheduler as L


class AbsScheduler(ABC):
    @abstractmethod
    def step(self, epoch: int = None):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state):
        pass


# If you need to define custom scheduler, please inherit these classes
class AbsBatchStepScheduler(AbsScheduler):
    @abstractmethod
    def step(self, epoch: int = None):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state):
        pass


class AbsEpochStepScheduler(AbsScheduler):
    @abstractmethod
    def step(self, epoch: int = None):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state):
        pass


class AbsValEpochStepScheduler(AbsEpochStepScheduler):
    @abstractmethod
    def step(self, val, epoch: int = None):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state):
        pass


# Create alias type to check the type
# Note(kamo): Currently PyTorch doesn't provide the base class
# to judge these classes.
AbsValEpochStepScheduler.register(L.ReduceLROnPlateau)
for s in [
    L.ReduceLROnPlateau,
    L.LambdaLR,
    L.StepLR,
    L.MultiStepLR,
    L.MultiStepLR,
    L.ExponentialLR,
    L.CosineAnnealingLR,
]:
    AbsEpochStepScheduler.register(s)

AbsBatchStepScheduler.register(L.CyclicLR)
for s in [
    L.OneCycleLR,
    L.CosineAnnealingWarmRestarts,
]:
    AbsBatchStepScheduler.register(s)
