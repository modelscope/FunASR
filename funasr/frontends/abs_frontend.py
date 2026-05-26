from abc import ABC, abstractmethod
import torch


class AbsFrontend(ABC, torch.nn.Module):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, input, input_lengths):
        raise NotImplementedError
