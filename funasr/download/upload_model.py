import os
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig

from modelscope.hub.api import HubApi


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):

    model_id = cfg.get("model_id", "FunAudioLLM/test")
    model_dir = cfg.get("model_dir")
    assert os.path.exists(
        model_dir
    ), f"{model_dir} does not exist!"  # 本地模型目录，要求目录中必须包含configuration.json

    # "TOKEN" '请从ModelScope个人中心->访问令牌获取'
    if "TOKEN" in os.environ:
        TOKEN = os.environ["TOKEN"]
    else:
        TOKEN = cfg.get("TOKEN", "")

    assert TOKEN is not None, f"{TOKEN} is None"

    api = HubApi()
    api.login(TOKEN)

    api.push_model(model_id=model_id, model_dir=model_dir)


if __name__ == "__main__":
    main_hydra()
