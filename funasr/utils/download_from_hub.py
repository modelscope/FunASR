import os
from omegaconf import OmegaConf
import torch
from funasr.utils.name_maps_from_hub import name_maps_ms, name_maps_hf

def download_model(**kwargs):
	model_hub = kwargs.get("model_hub", "ms")
	if model_hub == "ms":
		kwargs = download_fr_ms(**kwargs)
	
	return kwargs

def download_fr_ms(**kwargs):
	model_or_path = kwargs.get("model_pretrain")
	model_revision = kwargs.get("model_pretrain_revision")
	if not os.path.exists(model_or_path):
		model_or_path = get_or_download_model_dir(model_or_path, model_revision, third_party="funasr")
	
	config = os.path.join(model_or_path, "config.yaml")
	assert os.path.exists(config), "{} is not exist!".format(config)
	cfg = OmegaConf.load(config)
	kwargs = OmegaConf.merge(cfg, kwargs)
	init_param = os.path.join(model_or_path, "model.pb")
	kwargs["init_param"] = init_param
	kwargs["token_list"] = os.path.join(model_or_path, "tokens.txt")
	
	return kwargs

def get_or_download_model_dir(
                              model,
                              model_revision=None,
                              third_party=None):
	""" Get local model directory or download model if necessary.

	Args:
		model (str): model id or path to local model directory.
		model_revision  (str, optional): model version number.
		third_party (str, optional): in which third party library
			this function is called.
	"""
	from modelscope.hub.check_model import check_local_model_is_latest
	from modelscope.hub.snapshot_download import snapshot_download

	from modelscope.utils.constant import Invoke, ThirdParty
	
	if os.path.exists(model):
		model_cache_dir = model if os.path.isdir(
			model) else os.path.dirname(model)
		check_local_model_is_latest(
			model_cache_dir,
			user_agent={
				Invoke.KEY: Invoke.LOCAL_TRAINER,
				ThirdParty.KEY: third_party
			})
	else:
		model_cache_dir = snapshot_download(
			model,
			revision=model_revision,
			user_agent={
				Invoke.KEY: Invoke.TRAINER,
				ThirdParty.KEY: third_party
			})
	return model_cache_dir