import os
from omegaconf import OmegaConf
import torch
from funasr.download.name_maps_from_hub import name_maps_ms, name_maps_hf

def download_model(**kwargs):
	model_hub = kwargs.get("model_hub", "ms")
	if model_hub == "ms":
		kwargs = download_fr_ms(**kwargs)
	
	return kwargs

def download_fr_ms(**kwargs):
	model_or_path = kwargs.get("model")
	model_revision = kwargs.get("model_revision")
	if not os.path.exists(model_or_path):
		model_or_path = get_or_download_model_dir(model_or_path, model_revision, is_training=kwargs.get("is_training"))
	
	config = os.path.join(model_or_path, "config.yaml")
	assert os.path.exists(config), "{} is not exist!".format(config)
	cfg = OmegaConf.load(config)
	kwargs = OmegaConf.merge(cfg, kwargs)
	init_param = os.path.join(model_or_path, "model.pb")
	kwargs["init_param"] = init_param
	kwargs["token_list"] = os.path.join(model_or_path, "tokens.txt")
	kwargs["model"] = cfg["model"]
	kwargs["frontend_conf"]["cmvn_file"] = os.path.join(model_or_path, "am.mvn")
	
	return kwargs

def get_or_download_model_dir(
                              model,
                              model_revision=None,
							  is_training=False,
	):
	""" Get local model directory or download model if necessary.

	Args:
		model (str): model id or path to local model directory.
		model_revision  (str, optional): model version number.
		:param is_training:
	"""
	from modelscope.hub.check_model import check_local_model_is_latest
	from modelscope.hub.snapshot_download import snapshot_download

	from modelscope.utils.constant import Invoke, ThirdParty
	
	key = Invoke.LOCAL_TRAINER if is_training else Invoke.PIPELINE
	
	if os.path.exists(model):
		model_cache_dir = model if os.path.isdir(
			model) else os.path.dirname(model)
		check_local_model_is_latest(
			model_cache_dir,
			user_agent={
				Invoke.KEY: key,
				ThirdParty.KEY: "funasr"
			})
	else:
		model_cache_dir = snapshot_download(
			model,
			revision=model_revision,
			user_agent={
				Invoke.KEY: key,
				ThirdParty.KEY: "funasr"
			})
	return model_cache_dir