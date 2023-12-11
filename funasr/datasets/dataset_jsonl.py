import torch
import json
import torch.distributed as dist
import numpy as np
import kaldiio
import librosa
import torchaudio
import time
import logging

def load_audio(audio_path: str, fs: int=16000):
	audio = None
	if audio_path.startswith("oss:"):
		pass
	elif audio_path.startswith("odps:"):
		pass
	else:
		if ".ark:" in audio_path:
			audio = kaldiio.load_mat(audio_path)
		else:
			# audio, fs = librosa.load(audio_path, sr=fs)
			audio, fs = torchaudio.load(audio_path)
			audio = audio[0, :]
	return audio

def extract_features(data, date_type: str="sound", frontend=None):
	if date_type == "sound":

		if isinstance(data, np.ndarray):
			data = torch.from_numpy(data).to(torch.float32)
		data_len = torch.tensor([data.shape[0]]).to(torch.int32)
		feat, feats_lens = frontend(data[None, :], data_len)

		feat = feat[0, :, :]
	else:
		feat, feats_lens = torch.from_numpy(data).to(torch.float32), torch.tensor([data.shape[0]]).to(torch.int32)
	return feat, feats_lens
	
	

class IndexedDatasetJsonl(torch.utils.data.Dataset):
	
	def __init__(self, path):
		super().__init__()
		
		contents = []
		with open(path, encoding='utf-8') as fin:
			for line in fin:
				data = json.loads(line.strip())
				if "text" in data:  # for sft
					self.contents.append(data['text'])
				if "source" in data:  # for speech lab pretrain
					prompt = data["prompt"]
					source = data["source"]
					target = data["target"]
					source_len = data["source_len"]
					target_len = data["target_len"]

					contents.append({"source": source,
					                 "prompt": prompt,
					                 "target": target,
					                 "source_len": source_len,
					                 "target_len": target_len,
					                 }
					                )
		
		self.contents = []
		total_num = len(contents)
		try:
			rank = dist.get_rank()
			world_size = dist.get_world_size()
		except:
			rank = 0
			world_size = 1
			logging.warning("distributed is not initialized, only single shard")
		num_per_rank = total_num // world_size
		
		# rank = 0
		# import ipdb; ipdb.set_trace()
		self.contents = contents[rank * num_per_rank:(rank + 1) * num_per_rank]
	
		logging.info("in rank: {}, num of samplers: {}, total_num of samplers across ranks: {}".format(rank, len(self.contents), len(contents)))

	def __len__(self):
		return len(self.contents)
	
	def __getitem__(self, index):
		return self.contents[index]
	
	def get_source_len(self, data_dict):
		return data_dict["source_len"]

	def get_target_len(self, data_dict):
		
		return data_dict["target_len"] if "target_len" in data_dict else 0


class AudioDataset(torch.utils.data.Dataset):
	def __init__(self, path, frontend=None, tokenizer=None, int_pad_value: int = -1, float_pad_value: float = 0.0, **kwargs):
		super().__init__()
		self.indexed_dataset = IndexedDatasetJsonl(path)
		self.frontend = frontend.forward
		self.fs = 16000 if frontend is None else frontend.fs
		self.data_type = "sound"
		self.tokenizer = tokenizer

		self.int_pad_value = int_pad_value
		self.float_pad_value = float_pad_value

	

	
	def __len__(self):
		return len(self.indexed_dataset)
	
	def __getitem__(self, index):
		item = self.indexed_dataset[index]
		# return item

		source = item["source"]
		data_src = load_audio(source, fs=self.fs)
		speech, speech_lengths = extract_features(data_src, self.data_type, self.frontend)
		target = item["target"]
		ids = self.tokenizer.encode(target)
		ids_lengths = len(ids)
		text, text_lengths = torch.tensor(ids, dtype=torch.int64), torch.tensor([ids_lengths], dtype=torch.int32)

		return {"speech": speech,
		        "speech_lengths": speech_lengths,
		        "text": text,
		        "text_lengths": text_lengths,
		        }
	
	
	def collator(self, samples: list=None):
		
		# return samples
		
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

