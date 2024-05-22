import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf, ListConfig

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

    if kwargs.get("debug", False):
        import pdb

        pdb.set_trace()

    if "device" not in kwargs:
        kwargs["device"] = "cpu"
    model = AutoModel(**kwargs)

    res = model.export(
        input=kwargs.get("input", None),
        type=kwargs.get("type", "onnx"),
        quantize=kwargs.get("quantize", False),
        fallback_num=kwargs.get("fallback-num", 5),
        calib_num=kwargs.get("calib_num", 100),
        opset_version=kwargs.get("opset_version", 14),
    )
    print(res)


if __name__ == "__main__":
    main_hydra()
