import os.path

import torch
import numpy as np
import hydra
import json
from omegaconf import DictConfig, OmegaConf
from funasr.utils.dynamic_import import dynamic_import
import logging
from funasr.utils.download_from_hub import download_model
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.tokenizer.funtoken import build_tokenizer
from funasr.datasets.fun_datasets.load_audio_extract_fbank import load_bytes
from funasr.torch_utils.device_funcs import to_device
from tqdm import tqdm
from funasr.torch_utils.load_pretrained_model import load_pretrained_model
import time
import random
import string

@hydra.main(config_name=None, version_base=None)
def main_hydra(kwargs: DictConfig):
	assert "model" in kwargs

	pipeline = infer(**kwargs)
	res = pipeline(input=kwargs["input"])
	print(res)
	
def infer(**kwargs):
	
	if ":" not in kwargs["model"]:
		logging.info("download models from model hub: {}".format(kwargs.get("model_hub", "ms")))
		kwargs = download_model(**kwargs)
	
	set_all_random_seed(kwargs.get("seed", 0))

	
	device = kwargs.get("device", "cuda")
	if not torch.cuda.is_available() or kwargs.get("ngpu", 1):
		device = "cpu"
		batch_size = 1
	kwargs["device"] = device
	
	# build_tokenizer
	tokenizer = build_tokenizer(
		token_type=kwargs.get("token_type", "char"),
		bpemodel=kwargs.get("bpemodel", None),
		delimiter=kwargs.get("delimiter", None),
		space_symbol=kwargs.get("space_symbol", "<space>"),
		non_linguistic_symbols=kwargs.get("non_linguistic_symbols", None),
		g2p_type=kwargs.get("g2p_type", None),
		token_list=kwargs.get("token_list", None),
		unk_symbol=kwargs.get("unk_symbol", "<unk>"),
	)

	import pdb;
	pdb.set_trace()
	# build model
	model_class = dynamic_import(kwargs.get("model"))
	model = model_class(**kwargs, **kwargs["model_conf"], vocab_size=len(tokenizer.token_list))
	model.eval()
	model.to(device)
	frontend = model.frontend
	kwargs["token_list"] = tokenizer.token_list
	
	
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
	
	def _forward(input, input_len=None, **cfg):
		cfg = OmegaConf.merge(kwargs, cfg)
		date_type = cfg.get("date_type", "sound")
		
		key_list, data_list = build_iter_for_infer(input, input_len=input_len, date_type=date_type, frontend=frontend)
		
		speed_stats = {}
		asr_result_list = []
		num_samples = len(data_list)
		pbar = tqdm(colour="blue", total=num_samples, dynamic_ncols=True)
		for beg_idx in range(0, num_samples, batch_size):

			end_idx = min(num_samples, beg_idx + batch_size)
			data_batch = data_list[beg_idx:end_idx]
			key_batch = key_list[beg_idx:end_idx]
			batch = {"data_in": data_batch, "key": key_batch}
			
			time1 = time.perf_counter()
			results, meta_data = model.generate(**batch, tokenizer=tokenizer, **cfg)
			time2 = time.perf_counter()
			
			asr_result_list.append(results)
			pbar.update(1)
			
			# batch_data_time = time_per_frame_s * data_batch_i["speech_lengths"].sum().item()
			batch_data_time = meta_data.get("batch_data_time", -1)
			speed_stats["load_data"] = meta_data["load_data"]
			speed_stats["extract_feat"] = meta_data["extract_feat"]
			speed_stats["forward"] = f"{time2 - time1:0.3f}"
			speed_stats["rtf"] = f"{(time2 - time1)/batch_data_time:0.3f}"
			description = (
				f"{speed_stats}, "
			)
			pbar.set_description(description)
		
		torch.cuda.empty_cache()
		return asr_result_list
	
	return _forward
	

def build_iter_for_infer(data_in, input_len=None, date_type="sound", frontend=None):
	"""
	
	:param input:
	:param input_len:
	:param date_type:
	:param frontend:
	:return:
	"""
	data_list = []
	key_list = []
	filelist = [".scp", ".txt", ".json", ".jsonl"]
	
	chars = string.ascii_letters + string.digits
	
	if isinstance(data_in, str) and os.path.exists(data_in): # wav_pat; filelist: wav.scp, file.jsonl;text.txt;
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
	elif isinstance(data_in, (list, tuple)): # [audio sample point, fbank, wav_path]
		data_list = data_in
		key_list = ["rand_key_" + ''.join(random.choice(chars) for _ in range(13)) for _ in range(len(data_in))]
	else: # raw text; audio sample point, fbank
		if isinstance(data_in, bytes): # audio bytes
			data_in = load_bytes(data_in)
		key = "rand_key_" + ''.join(random.choice(chars) for _ in range(13))
		data_list = [data_in]
		key_list = [key]
	
	return key_list, data_list


if __name__ == '__main__':
	main_hydra()