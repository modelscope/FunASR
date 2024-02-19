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
	def __init__(self, speed_perturb: list=None, **kwargs):
		super().__init__()
		self.speed_perturb = speed_perturb
		
	def forward(self, waveform, fs, **kwargs):
		if self.speed_perturb is None:
			return waveform
		speed = random.choice(self.speed_perturb)
		if speed != 1.0:
			waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
				torch.tensor(waveform).view(1, -1), fs, [['speed', str(speed)], ['rate', str(fs)]])
			waveform = waveform.view(-1)
			
		return waveform


@tables.register("preprocessor_classes", "TextPreprocessSegDict")
class TextPreprocessSegDict(nn.Module):
	def __init__(self, seg_dict: str = None,
	             text_cleaner: Collection[str] = None,
	             split_with_space: bool = False,
	             **kwargs):
		super().__init__()
		
		self.seg_dict = None
		if seg_dict is not None:
			self.seg_dict = {}
			with open(seg_dict, "r", encoding="utf8") as f:
				lines = f.readlines()
			for line in lines:
				s = line.strip().split()
				key = s[0]
				value = s[1:]
				self.seg_dict[key] = " ".join(value)
		self.text_cleaner = TextCleaner(text_cleaner)
		self.split_with_space = split_with_space
	
	def forward(self, text, **kwargs):
		if self.seg_dict is not None:
			text = self.text_cleaner(text)
			if self.split_with_space:
				tokens = text.strip().split(" ")
				if self.seg_dict is not None:
					text = seg_tokenize(tokens, self.seg_dict)

		return text

def seg_tokenize(txt, seg_dict):
	pattern = re.compile(r'^[\u4E00-\u9FA50-9]+$')
	out_txt = ""
	for word in txt:
		word = word.lower()
		if word in seg_dict:
			out_txt += seg_dict[word] + " "
		else:
			if pattern.match(word):
				for char in word:
					if char in seg_dict:
						out_txt += seg_dict[char] + " "
					else:
						out_txt += "<unk>" + " "
			else:
				out_txt += "<unk>" + " "
	return out_txt.strip().split()