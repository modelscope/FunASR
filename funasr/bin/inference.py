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
    model = AutoModel(**kwargs)
    res = model.generate(input=kwargs["input"])
    print(res)


if __name__ == "__main__":
    main_hydra()
