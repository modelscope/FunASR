import os
import json
import torch
import logging
import concurrent.futures
import librosa
import torch.distributed as dist
from typing import Collection
import torch
import torchaudio
from torch import nn
import random
import re
from funasr.tokenizer.cleaner import TextCleaner
from funasr.register import tables


@tables.register("preprocessor_classes", "SpeechPreprocessSpeedPerturb")
class SpeechPreprocessSpeedPerturb(nn.Module):
    def __init__(self, speed_perturb: list = None, **kwargs):
        super().__init__()
        self.speed_perturb = speed_perturb

    def forward(self, waveform, fs, **kwargs):
        if self.speed_perturb is None:
            return waveform
        speed = random.choice(self.speed_perturb)
        if speed != 1.0:
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform)
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform.view(1, -1), fs, [["speed", str(speed)], ["rate", str(fs)]]
            )
            waveform = waveform.view(-1)

        return waveform


@tables.register("preprocessor_classes", "TextPreprocessSegDict")
class TextPreprocessSegDict(nn.Module):
    def __init__(
        self,
        seg_dict: str = None,
        text_cleaner: Collection[str] = None,
        split_with_space: bool = False,
        **kwargs
    ):
        super().__init__()

        self.text_cleaner = TextCleaner(text_cleaner)

    def forward(self, text, **kwargs):
        text = self.text_cleaner(text)

        return text
