import torch
import json
import torch.distributed as dist
import numpy as np
import kaldiio
import librosa
import torchaudio
import time
import logging

from funasr.datasets.audio_datasets.load_audio_extract_fbank import load_audio, extract_fbank
from funasr.utils.register import register_class, registry_tables

@register_class("dataset_classes", "AudioDataset")
class AudioDataset(torch.utils.data.Dataset):
	def __init__(self,
	             path,
	             index_ds: str = None,
	             frontend=None,
	             tokenizer=None,
	             int_pad_value: int = -1,
	             float_pad_value: float = 0.0,
	              **kwargs):
		super().__init__()
		index_ds_class = registry_tables.index_ds_classes.get(index_ds.lower())
		self.index_ds = index_ds_class(path)
		self.frontend = frontend
		self.fs = 16000 if frontend is None else frontend.fs
		self.data_type = "sound"
		self.tokenizer = tokenizer

		self.int_pad_value = int_pad_value
		self.float_pad_value = float_pad_value
	
	def get_source_len(self, index):
		item = self.index_ds[index]
		return self.index_ds.get_source_len(item)
	
	def get_target_len(self, index):
		item = self.index_ds[index]
		return self.index_ds.get_target_len(item)
	
	def __len__(self):
		return len(self.index_ds)
	
	def __getitem__(self, index):
		item = self.index_ds[index]
		# import pdb;
		# pdb.set_trace()
		source = item["source"]
		data_src = load_audio(source, fs=self.fs)
		speech, speech_lengths = extract_fbank(data_src, data_type=self.data_type, frontend=self.frontend) # speech: [b, T, d]
		target = item["target"]
		ids = self.tokenizer.encode(target)
		ids_lengths = len(ids)
		text, text_lengths = torch.tensor(ids, dtype=torch.int64), torch.tensor([ids_lengths], dtype=torch.int32)

		return {"speech": speech[0, :, :],
		        "speech_lengths": speech_lengths,
		        "text": text,
		        "text_lengths": text_lengths,
		        }
	
	
	def collator(self, samples: list=None):


		outputs = {}
		for sample in samples:
			for key in sample.keys():
				if key not in outputs:
					outputs[key] = []
				outputs[key].append(sample[key])

		for key, data_list in outputs.items():
			if data_list[0].dtype == torch.int64:

				pad_value = self.int_pad_value
			else:
				pad_value = self.float_pad_value
			outputs[key] = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=pad_value)
		return outputs

