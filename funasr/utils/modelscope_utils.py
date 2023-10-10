import os
from modelscope.hub.snapshot_download import snapshot_download
from pathlib import Path


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

def get_default_cache_dir():
    """
    default base dir: '~/.cache/modelscope'
    """
    default_cache_dir = Path.home().joinpath('.cache', 'modelscope')
    return default_cache_dir

def get_cache_dir(model_id):
    """cache dir precedence:
        function parameter > environment > ~/.cache/modelscope/hub

    Args:
        model_id (str, optional): The model id.

    Returns:
        str: the model_id dir if model_id not None, otherwise cache root dir.
    """
    default_cache_dir = get_default_cache_dir()
    base_path = os.getenv('MODELSCOPE_CACHE',
                          os.path.join(default_cache_dir, 'hub'))
    return base_path if model_id is None else os.path.join(
        base_path, model_id + '/')