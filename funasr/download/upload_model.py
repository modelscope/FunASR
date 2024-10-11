import os
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import traceback
from modelscope.hub.api import HubApi
from modelscope.hub.constants import (
    API_HTTP_CLIENT_TIMEOUT,
    API_RESPONSE_FIELD_DATA,
    API_RESPONSE_FIELD_EMAIL,
    API_RESPONSE_FIELD_GIT_ACCESS_TOKEN,
    API_RESPONSE_FIELD_MESSAGE,
    API_RESPONSE_FIELD_USERNAME,
    DEFAULT_CREDENTIALS_PATH,
    MODELSCOPE_CLOUD_ENVIRONMENT,
    MODELSCOPE_CLOUD_USERNAME,
    MODELSCOPE_REQUEST_ID,
    ONE_YEAR_SECONDS,
    REQUESTS_API_HTTP_METHOD,
    DatasetVisibility,
    Licenses,
    ModelVisibility,
)


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

    visibility = cfg.get("visibility", "PRIVATE")  # PRIVATE #ModelVisibility.PUBLIC
    visibility = getattr(ModelVisibility, visibility)
    model_name = cfg.get("model_name", "测试")

    api = HubApi()
    api.login(TOKEN)

    try:
        api.create_model(
            model_id=model_id,
            visibility=visibility,
            license=Licenses.APACHE_V2,
            chinese_name=model_name,
        )
    except Exception as e:
        print(f"Create_model failed! {str(e)}, {traceback.format_exc()}")

    print(f"model url: https://modelscope.cn/models/{model_id}")

    api.push_model(model_id=model_id, model_dir=model_dir)
    print(
        f"Upload model finished."
        f"model_dir: {model_dir}"
        f"model_id: {model_id}"
        f"url: https://modelscope.cn/models/{model_id}"
    )


"""
TOKEN="fadd1abb-4df6-4807-9051-5ab01ac81071"
model_id="iic/Whisper-large-v3-turbo"
model_dir="/Users/zhifu/Downloads/Whisper-large-v3-turbo"

python -m funasr.download.upload_model ++TOKEN=${TOKEN} ++model_id=${model_id} ++model_dir=${model_dir}
"""

if __name__ == "__main__":
    main_hydra()