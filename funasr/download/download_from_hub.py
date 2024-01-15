import os
import json
from omegaconf import OmegaConf

from funasr.download.name_maps_from_hub import name_maps_ms, name_maps_hf


def download_model(**kwargs):
    model_hub = kwargs.get("model_hub", "ms")
    if model_hub == "ms":
        kwargs = download_from_ms(**kwargs)
    
    return kwargs

def download_from_ms(**kwargs):
    model_or_path = kwargs.get("model")
    if model_or_path in name_maps_ms:
        model_or_path = name_maps_ms[model_or_path]
    model_revision = kwargs.get("model_revision")
    if not os.path.exists(model_or_path):
        model_or_path = get_or_download_model_dir(model_or_path, model_revision, is_training=kwargs.get("is_training"), check_latest=kwargs.get("kwargs", True))
    kwargs["model_path"] = model_or_path
    
    config = os.path.join(model_or_path, "config.yaml")
    if os.path.exists(config) and os.path.exists(os.path.join(model_or_path, "model.pb")):
        
        config = OmegaConf.load(config)
        kwargs = OmegaConf.merge(config, kwargs)
        init_param = os.path.join(model_or_path, "model.pb")
        kwargs["init_param"] = init_param
        if os.path.exists(os.path.join(model_or_path, "tokens.txt")):
            kwargs["tokenizer_conf"]["token_list"] = os.path.join(model_or_path, "tokens.txt")
        if os.path.exists(os.path.join(model_or_path, "tokens.json")):
            kwargs["tokenizer_conf"]["token_list"] = os.path.join(model_or_path, "tokens.json")
        if os.path.exists(os.path.join(model_or_path, "seg_dict")):
            kwargs["tokenizer_conf"]["seg_dict"] = os.path.join(model_or_path, "seg_dict")
        if os.path.exists(os.path.join(model_or_path, "bpe.model")):
            kwargs["tokenizer_conf"]["bpemodel"] = os.path.join(model_or_path, "bpe.model")
        kwargs["model"] = config["model"]
        if os.path.exists(os.path.join(model_or_path, "am.mvn")):
            kwargs["frontend_conf"]["cmvn_file"] = os.path.join(model_or_path, "am.mvn")
        if os.path.exists(os.path.join(model_or_path, "jieba_usr_dict")):
            kwargs["jieba_usr_dict"] = os.path.join(model_or_path, "jieba_usr_dict")
    elif os.path.exists(os.path.join(model_or_path, "configuration.json")):
        with open(os.path.join(model_or_path, "configuration.json"), 'r', encoding='utf-8') as f:
            conf_json = json.load(f)
            cfg = {}
            add_file_root_path(model_or_path, conf_json["file_path_metas"], cfg)
            cfg.update(kwargs)
            config = OmegaConf.load(cfg["config"])
            kwargs = OmegaConf.merge(config, cfg)
        kwargs["model"] = config["model"]
    return OmegaConf.to_container(kwargs, resolve=True)

def add_file_root_path(model_or_path: str, file_path_metas: dict, cfg = {}):
    
    if isinstance(file_path_metas, dict):
        for k, v in file_path_metas.items():
            if isinstance(v, str):
                p = os.path.join(model_or_path, v)
                if os.path.exists(p):
                    cfg[k] = p
            elif isinstance(v, dict):
                if k not in cfg:
                    cfg[k] = {}
                return add_file_root_path(model_or_path, v, cfg[k])
    
    return cfg


def get_or_download_model_dir(
        model,
        model_revision=None,
        is_training=False,
        check_latest=True,
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
    
    if os.path.exists(model) and check_latest:
        model_cache_dir = model if os.path.isdir(
            model) else os.path.dirname(model)
        try:
            check_local_model_is_latest(
                model_cache_dir,
                user_agent={
                    Invoke.KEY: key,
                    ThirdParty.KEY: "funasr"
                })
        except:
            print("could not check the latest version")
    else:
        model_cache_dir = snapshot_download(
            model,
            revision=model_revision,
            user_agent={
                Invoke.KEY: key,
                ThirdParty.KEY: "funasr"
            })
    return model_cache_dir