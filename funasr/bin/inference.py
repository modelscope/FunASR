import json
import time
import torch
import hydra
import random
import string
import logging
import os.path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, ListConfig

from funasr.register import tables
from funasr.utils.load_utils import load_bytes
from funasr.download.file import download_from_url
from funasr.download.download_from_hub import download_model
from funasr.utils.vad_utils import slice_padding_audio_samples
from funasr.train_utils.set_all_random_seed import set_all_random_seed
from funasr.train_utils.load_pretrained_model import load_pretrained_model
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils.timestamp_tools import timestamp_sentence
from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
from funasr.models.campplus.cluster_backend import ClusterBackend
from funasr.auto.auto_model import AutoModel


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    def to_plain_list(cfg_item):
        if isinstance(cfg_item, ListConfig):
            return OmegaConf.to_container(cfg_item, resolve=True)
        elif isinstance(cfg_item, DictConfig):
            return {k: to_plain_list(v) for k, v in cfg_item.items()}
        else:
            return cfg_item
    
    kwargs = to_plain_list(cfg)
    log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())

    logging.basicConfig(level=log_level)

    if kwargs.get("debug", False):
        import pdb; pdb.set_trace()
    model = AutoModel(**kwargs)
    res = model(input=kwargs["input"])
    print(res)



if __name__ == '__main__':
    main_hydra()