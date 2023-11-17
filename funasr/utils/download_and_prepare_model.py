
import os
from pathlib import Path
import logging

name_maps_ms = {
    "paraformer-zh": "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-zh-spk": "damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn",
    "paraformer-en": "damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
    "paraformer-en-spk": "damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
    "paraformer-zh-streaming": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "fsmn-vad": "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "damo/punc_ct-transformer_cn-en-common-vocab471067-large",
    "fa-zh": "damo/speech_timestamp_prediction-v1-16k-offline",
}

def prepare_model(
	model: str = None,
	# mode: str = None,
	vad_model: str = None,
	punc_model: str = None,
	model_hub: str = "ms",
	cache_dir: str = None,
	**kwargs,
):
	if not Path(model).exists():
		if model_hub == "ms" or model_hub == "modelscope":
			from modelscope.utils.logger import get_logger
			
			logger = get_logger(log_level=logging.CRITICAL)
			logger.setLevel(logging.CRITICAL)
			try:
				from modelscope.hub.snapshot_download import snapshot_download as download_tool
				model = name_maps_ms[model] if model is not None else None
				vad_model = name_maps_ms[vad_model] if vad_model is not None else None
				punc_model = name_maps_ms[punc_model] if punc_model is not None else None
			except:
				raise "You are exporting model from modelscope, please install modelscope and try it again. To install modelscope, you could:\n" \
				      "\npip3 install -U modelscope\n" \
				      "For the users in China, you could install with the command:\n" \
				      "\npip3 install -U modelscope -i https://mirror.sjtu.edu.cn/pypi/web/simple"

			try:
				model = download_tool(model, cache_dir=cache_dir, revision=kwargs.get("revision", None))
				print("asr model have been downloaded to: {}".format(model))
			except:
				raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(
					model)
		
		elif model_hub == "hf" or model_hub == "huggingface":
			download_tool = 0
		else:
			raise "model_hub must be on of ms or hf, but get {}".format(model_hub)

		
		if vad_model is not None and not Path(vad_model).exists():
			vad_model = download_tool(vad_model, cache_dir=cache_dir)
			print("vad_model have been downloaded to: {}".format(vad_model))
		if punc_model is not None and not Path(punc_model).exists():
			punc_model = download_tool(punc_model, cache_dir=cache_dir)
			print("punc_model have been downloaded to: {}".format(punc_model))
		
		# asr
		kwargs.update({"cmvn_file": None if model is None else os.path.join(model, "am.mvn"),
		               "asr_model_file": None if model is None else os.path.join(model, "model.pb"),
		               "asr_train_config": None if model is None else os.path.join(model, "config.yaml"),
		               })
		mode = kwargs.get("mode", None)
		if mode is None:
			import json
			json_file = os.path.join(model, 'configuration.json')
			with open(json_file, 'r') as f:
				config_data = json.load(f)
				if config_data['task'] == "punctuation":
					mode = config_data['model']['punc_model_config']['mode']
				else:
					mode = config_data['model']['model_config']['mode']
		if vad_model is not None and "vad" not in mode:
			mode = "paraformer_vad"
		kwargs["mode"] = mode
		# vad
		kwargs.update({"vad_cmvn_file": None if vad_model is None else os.path.join(vad_model, "vad.mvn"),
		               "vad_model_file": None if vad_model is None else os.path.join(vad_model, "vad.pb"),
		               "vad_infer_config": None if vad_model is None else os.path.join(vad_model, "vad.yaml"),
		               })
		# punc
		kwargs.update({
			"punc_model_file": None if punc_model is None else os.path.join(punc_model, "punc.pb"),
			"punc_infer_config": None if punc_model is None else os.path.join(punc_model, "punc.yaml"),
		})
		
		
		return model, vad_model, punc_model, kwargs
