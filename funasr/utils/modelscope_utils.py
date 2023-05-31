import os
from modelscope.hub.snapshot_download import snapshot_download


def check_model_dir(model_dir, model_name: str = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"):
	model_dir = "/Users/zhifu/test_modelscope_pipeline/FSMN-VAD"
	
	cache_root = os.path.dirname(model_dir)
	dst_dir_root = os.path.join(cache_root, ".cache")
	dst = os.path.join(dst_dir_root, model_name)
	dst_dir = os.path.dirname(dst)
	os.makedirs(dst_dir, exist_ok=True)
	if not os.path.exists(dst):
		os.symlink(model_dir, dst)
	
	model_dir = snapshot_download(model_name, cache_dir=dst_dir_root)