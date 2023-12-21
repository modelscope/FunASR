import os.path

import torch
import numpy as np
import hydra
import json
from omegaconf import DictConfig, OmegaConf
import logging
from funasr.download.download_from_hub import download_model
from funasr.train_utils.set_all_random_seed import set_all_random_seed
from funasr.datasets.audio_datasets.load_audio_extract_fbank import load_bytes
from funasr.train_utils.device_funcs import to_device
from tqdm import tqdm
from funasr.train_utils.load_pretrained_model import load_pretrained_model
import time
import random
import string
from funasr.register import tables
from funasr.datasets.audio_datasets.load_audio_extract_fbank import load_audio
from funasr.utils.vad_utils import slice_padding_audio_samples

def build_iter_for_infer(data_in, input_len=None, data_type="sound"):
	"""
	
	:param input:
	:param input_len:
	:param data_type:
	:param frontend:
	:return:
	"""
	data_list = []
	key_list = []
	filelist = [".scp", ".txt", ".json", ".jsonl"]
	
	chars = string.ascii_letters + string.digits
	
	if isinstance(data_in, str) and os.path.exists(data_in): # wav_path; filelist: wav.scp, file.jsonl;text.txt;
		_, file_extension = os.path.splitext(data_in)
		file_extension = file_extension.lower()
		if file_extension in filelist: #filelist: wav.scp, file.jsonl;text.txt;
			with open(data_in, encoding='utf-8') as fin:
				for line in fin:
					key = "rand_key_" + ''.join(random.choice(chars) for _ in range(13))
					if data_in.endswith(".jsonl"): #file.jsonl: json.dumps({"source": data})
						lines = json.loads(line.strip())
						data = lines["source"]
						key = data["key"] if "key" in data else key
					else: # filelist, wav.scp, text.txt: id \t data or data
						lines = line.strip().split()
						data = lines[1] if len(lines)>1 else lines[0]
						key = lines[0] if len(lines)>1 else key
					
					data_list.append(data)
					key_list.append(key)
		else:
			key = "rand_key_" + ''.join(random.choice(chars) for _ in range(13))
			data_list = [data_in]
			key_list = [key]
	elif isinstance(data_in, (list, tuple)): # [audio sample point, fbank]
		data_list = data_in
		key_list = ["rand_key_" + ''.join(random.choice(chars) for _ in range(13)) for _ in range(len(data_in))]
	else: # raw text; audio sample point, fbank; bytes
		if isinstance(data_in, bytes): # audio bytes
			data_in = load_bytes(data_in)
		key = "rand_key_" + ''.join(random.choice(chars) for _ in range(13))
		data_list = [data_in]
		key_list = [key]
	
	return key_list, data_list

@hydra.main(config_name=None, version_base=None)
def main_hydra(kwargs: DictConfig):
	log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())

	logging.basicConfig(level=log_level)

	if kwargs.get("debug", False):
		import pdb; pdb.set_trace()
	model = AutoModel(**kwargs)
	res = model(input=kwargs["input"])
	print(res)

class AutoModel:
	
	def __init__(self, **kwargs):
		tables.print()
		
		model, kwargs = self.build_model(**kwargs)
		
		# if vad_model is not None, build vad model else None
		vad_model = kwargs.get("vad_model", None)
		vad_kwargs = kwargs.get("vad_model_revision", None)
		if vad_model is not None:
			print("build vad model")
			vad_kwargs = {"model": vad_model, "model_revision": vad_kwargs}
			vad_model, vad_kwargs = self.build_model(**vad_kwargs)

		# if punc_model is not None, build punc model else None
		punc_model = kwargs.get("punc_model", None)
		punc_kwargs = kwargs.get("punc_model_revision", None)
		if punc_model is not None:
			punc_kwargs = {"model": punc_model, "model_revision": punc_kwargs}
			punc_model, punc_kwargs = self.build_model(**punc_kwargs)
			
		self.kwargs = kwargs
		self.model = model
		self.vad_model = vad_model
		self.vad_kwargs = vad_kwargs
		self.punc_model = punc_model
		self.punc_kwargs = punc_kwargs
		
		

	def build_model(self, **kwargs):
		assert "model" in kwargs
		if "model_conf" not in kwargs:
			logging.info("download models from model hub: {}".format(kwargs.get("model_hub", "ms")))
			kwargs = download_model(**kwargs)
		
		set_all_random_seed(kwargs.get("seed", 0))
		
		device = kwargs.get("device", "cuda")
		if not torch.cuda.is_available() or kwargs.get("ngpu", 1):
			device = "cpu"
			kwargs["batch_size"] = 1
		kwargs["device"] = device
		
		# build tokenizer
		tokenizer = kwargs.get("tokenizer", None)
		if tokenizer is not None:
			tokenizer_class = tables.tokenizer_classes.get(tokenizer.lower())
			tokenizer = tokenizer_class(**kwargs["tokenizer_conf"])
			kwargs["tokenizer"] = tokenizer
			kwargs["token_list"] = tokenizer.token_list
		
		# build frontend
		frontend = kwargs.get("frontend", None)
		if frontend is not None:
			frontend_class = tables.frontend_classes.get(frontend.lower())
			frontend = frontend_class(**kwargs["frontend_conf"])
			kwargs["frontend"] = frontend
			kwargs["input_size"] = frontend.output_size()
		
		# build model
		model_class = tables.model_classes.get(kwargs["model"].lower())
		model = model_class(**kwargs, **kwargs["model_conf"],
		                    vocab_size=len(tokenizer.token_list) if tokenizer is not None else -1)
		model.eval()
		model.to(device)
		
		# init_param
		init_param = kwargs.get("init_param", None)
		if init_param is not None:
			logging.info(f"Loading pretrained params from {init_param}")
			load_pretrained_model(
				model=model,
				init_param=init_param,
				ignore_init_mismatch=kwargs.get("ignore_init_mismatch", False),
				oss_bucket=kwargs.get("oss_bucket", None),
			)
		
		return model, kwargs
	
	def __call__(self, input, input_len=None, **cfg):
		if self.vad_model is None:
			return self.generate(input, input_len=input_len, **cfg)
			
		else:
			return self.generate_with_vad(input, input_len=input_len, **cfg)
		
	def generate(self, input, input_len=None, model=None, kwargs=None, **cfg):
		kwargs = self.kwargs if kwargs is None else kwargs
		kwargs.update(cfg)
		model = self.model if model is None else model
		
		data_type = kwargs.get("data_type", "sound")
		batch_size = kwargs.get("batch_size", 1)
		# if kwargs.get("device", "cpu") == "cpu":
		# 	batch_size = 1
		
		key_list, data_list = build_iter_for_infer(input, input_len=input_len, data_type=data_type)
		
		speed_stats = {}
		asr_result_list = []
		num_samples = len(data_list)
		pbar = tqdm(colour="blue", total=num_samples+1, dynamic_ncols=True)
		time_speech_total = 0.0
		time_escape_total = 0.0
		for beg_idx in range(0, num_samples, batch_size):
			end_idx = min(num_samples, beg_idx + batch_size)
			data_batch = data_list[beg_idx:end_idx]
			key_batch = key_list[beg_idx:end_idx]
			batch = {"data_in": data_batch, "key": key_batch}
			if (end_idx - beg_idx) == 1 and isinstance(data_batch[0], torch.Tensor): # fbank
				batch["data_batch"] = data_batch[0]
				batch["data_lengths"] = input_len
		
			time1 = time.perf_counter()
			results, meta_data = model.generate(**batch, **kwargs)
			time2 = time.perf_counter()
			
			asr_result_list.extend(results)
			pbar.update(1)
			
			# batch_data_time = time_per_frame_s * data_batch_i["speech_lengths"].sum().item()
			batch_data_time = meta_data.get("batch_data_time", -1)
			time_escape = time2 - time1
			speed_stats["load_data"] = meta_data.get("load_data", 0.0)
			speed_stats["extract_feat"] = meta_data.get("extract_feat", 0.0)
			speed_stats["forward"] = f"{time_escape:0.3f}"
			speed_stats["batch_size"] = f"{len(results)}"
			speed_stats["rtf"] = f"{(time_escape) / batch_data_time:0.3f}"
			description = (
				f"{speed_stats}, "
			)
			pbar.set_description(description)
			time_speech_total += batch_data_time
			time_escape_total += time_escape
			
		pbar.update(1)
		pbar.set_description(f"rtf_avg: {time_escape_total/time_speech_total:0.3f}")
		torch.cuda.empty_cache()
		return asr_result_list
	
	def generate_with_vad(self, input, input_len=None, **cfg):
		
		# step.1: compute the vad model
		model = self.vad_model
		kwargs = self.vad_kwargs
		beg_vad = time.time()
		res = self.generate(input, input_len=input_len, model=model, kwargs=kwargs, **cfg)
		end_vad = time.time()
		print(f"time cost vad: {end_vad - beg_vad:0.3f}")


		# step.2 compute asr model
		model = self.model
		kwargs = self.kwargs
		kwargs.update(cfg)
		batch_size = int(kwargs.get("batch_size_s", 300))*1000
		batch_size_threshold_ms = int(kwargs.get("batch_size_threshold_s", 60))*1000
		kwargs["batch_size"] = batch_size
		data_type = kwargs.get("data_type", "sound")
		key_list, data_list = build_iter_for_infer(input, input_len=input_len, data_type=data_type)
		results_ret_list = []
		time_speech_total_all_samples = 0.0

		beg_total = time.time()
		pbar_total = tqdm(colour="red", total=len(res) + 1, dynamic_ncols=True)
		for i in range(len(res)):
			key = res[i]["key"]
			vadsegments = res[i]["value"]
			input_i = data_list[i]
			speech = load_audio(input_i, fs=kwargs["frontend"].fs, audio_fs=kwargs.get("fs", 16000))
			speech_lengths = len(speech)
			n = len(vadsegments)
			data_with_index = [(vadsegments[i], i) for i in range(n)]
			sorted_data = sorted(data_with_index, key=lambda x: x[0][1] - x[0][0])
			results_sorted = []
			
			if not len(sorted_data):
				logging.info("decoding, utt: {}, empty speech".format(key))
				continue
			

			# if kwargs["device"] == "cpu":
			# 	batch_size = 0
			if len(sorted_data) > 0 and len(sorted_data[0]) > 0:
				batch_size = max(batch_size, sorted_data[0][0][1] - sorted_data[0][0][0])
			
			batch_size_ms_cum = 0
			beg_idx = 0
			beg_asr_total = time.time()
			time_speech_total_per_sample = speech_lengths/16000
			time_speech_total_all_samples += time_speech_total_per_sample

			for j, _ in enumerate(range(0, n)):
				batch_size_ms_cum += (sorted_data[j][0][1] - sorted_data[j][0][0])
				if j < n - 1 and (
					batch_size_ms_cum + sorted_data[j + 1][0][1] - sorted_data[j + 1][0][0]) < batch_size and (
					sorted_data[j + 1][0][1] - sorted_data[j + 1][0][0]) < batch_size_threshold_ms:
					continue
				batch_size_ms_cum = 0
				end_idx = j + 1
				speech_j, speech_lengths_j = slice_padding_audio_samples(speech, speech_lengths, sorted_data[beg_idx:end_idx])
				beg_idx = end_idx

				results = self.generate(speech_j, input_len=None, model=model, kwargs=kwargs, **cfg)
	
				if len(results) < 1:
					continue
				results_sorted.extend(results)


			pbar_total.update(1)
			end_asr_total = time.time()
			time_escape_total_per_sample = end_asr_total - beg_asr_total
			pbar_total.set_description(f"rtf_avg_per_sample: {time_escape_total_per_sample / time_speech_total_per_sample:0.3f}, "
			                     f"time_speech_total_per_sample: {time_speech_total_per_sample: 0.3f}, "
			                     f"time_escape_total_per_sample: {time_escape_total_per_sample:0.3f}")

			restored_data = [0] * n
			for j in range(n):
				index = sorted_data[j][1]
				restored_data[index] = results_sorted[j]
			result = {}
			
			for j in range(n):
				for k, v in restored_data[j].items():
					if not k.startswith("timestamp"):
						if k not in result:
							result[k] = restored_data[j][k]
						else:
							result[k] += restored_data[j][k]
					else:
						result[k] = []
						for t in restored_data[j][k]:
							t[0] += vadsegments[j][0]
							t[1] += vadsegments[j][0]
						result[k] += restored_data[j][k]
						
			result["key"] = key
			results_ret_list.append(result)
			pbar_total.update(1)
		pbar_total.update(1)
		end_total = time.time()
		time_escape_total_all_samples = end_total - beg_total
		pbar_total.set_description(f"rtf_avg_all_samples: {time_escape_total_all_samples / time_speech_total_all_samples:0.3f}, "
		                     f"time_speech_total_all_samples: {time_speech_total_all_samples: 0.3f}, "
		                     f"time_escape_total_all_samples: {time_escape_total_all_samples:0.3f}")
		return results_ret_list
	
if __name__ == '__main__':
	main_hydra()