import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def download_module(monkeypatch):
    # Load the downloader without importing FunASR's optional model dependencies.
    download_dir = Path(__file__).resolve().parent.parent / "funasr" / "download"
    for name in ("name_maps_from_hub", "download_model_from_hub"):
        module_name = f"funasr.download.{name}"
        spec = importlib.util.spec_from_file_location(module_name, download_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)

    def fail_download(*args, **kwargs):
        pytest.fail("Local model loading must not download from HuggingFace")

    monkeypatch.setattr(module, "get_or_download_model_dir_hf", fail_download)
    return module


@pytest.mark.parametrize("explicit_model_path", [True, False], ids=["model_path", "local_model"])
def test_download_from_hf_loads_local_config(tmp_path, download_module, explicit_model_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("model: Paraformer\nmodel_conf: {}\n", encoding="utf-8")
    (model_dir / "model.pt").touch()
    kwargs = (
        {"model": "example/asr", "model_path": str(model_dir)}
        if explicit_model_path
        else {"model": str(model_dir)}
    )

    result = download_module.download_model(**kwargs, hub="hf")

    assert result["model"] == "Paraformer"
    assert result["init_param"] == str(model_dir / "model.pt")
