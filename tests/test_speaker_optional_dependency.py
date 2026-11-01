"""Exercise optional speaker imports in fresh processes without model downloads."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


_PROBE = r"""
import importlib.abc
import importlib.util
import sys
from unittest.mock import patch

module_name, failure, model_dir = sys.argv[1:]
if failure == "import":
    original_error = ModuleNotFoundError(
        "No module named 'speaker_test_dependency'", name="speaker_test_dependency"
    )
else:
    original_error = RuntimeError("speaker dependency initialization failed")


class BrokenSpeakerImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == module_name:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise original_error


if module_name:
    sys.meta_path.insert(0, BrokenSpeakerImport())

import torch
from funasr import AutoModel
from funasr.register import tables

built_models = []


class DummyModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        built_models.append(kwargs["model"])


tables.model_classes["speaker-import-test"] = DummyModel
tables.model_classes["cam++"] = DummyModel
config = dict(
    model="speaker-import-test", model_conf={}, device="cpu", ncpu=1, disable_update=True
)

speaker_config = dict(config)
if model_dir:
    speaker_config["model"] = model_dir
    del speaker_config["model_conf"]
else:
    speaker_config.update(
        spk_model="cam++",
        spk_kwargs={"model_conf": {}, "cb_kwargs": {"merge_thr": 0.5}},
    )

with patch(
    "funasr.download.download_model_from_hub.get_or_download_model_dir",
    side_effect=AssertionError("model hub access"),
) as download:
    asr = AutoModel(**config)
    assert isinstance(asr.model, DummyModel)
    assert asr.spk_model is None
    print("ASR-only construction: OK")

    if module_name:
        try:
            AutoModel(**speaker_config)
        except ImportError as error:
            assert "spk_model" in str(error), str(error)
            assert "dependencies" in str(error), str(error)
            assert "install" in str(error).lower(), str(error)
            assert error.__cause__ is original_error, repr(error.__cause__)
            assert "cam++" not in built_models, built_models
            print("Speaker construction: actionable ImportError with original cause")
        else:
            raise AssertionError("Speaker construction should report the import failure")
    else:
        from funasr.models.campplus.cluster_backend import ClusterBackend

        speaker = AutoModel(**speaker_config)
        assert isinstance(speaker.spk_model, DummyModel)
        assert isinstance(speaker.cb_model, ClusterBackend)
        assert speaker.cb_model.model_config["merge_thr"] == 0.5
        print("Speaker construction with available dependencies: OK")

    download.assert_not_called()
"""


@pytest.fixture(params=["kwargs", "config"])
def speaker_config(request, tmp_path):
    if request.param == "kwargs":
        return ""

    (tmp_path / "configuration.json").write_text(
        '{"file_path_metas": {"config": "config.yaml"}}', encoding="utf-8"
    )
    (tmp_path / "config.yaml").write_text(
        "model: speaker-import-test\n"
        "model_conf: {}\n"
        "spk_model: cam++\n"
        "spk_kwargs:\n"
        "  model_conf: {}\n"
        "  cb_kwargs:\n"
        "    merge_thr: 0.5\n",
        encoding="utf-8",
    )
    return str(tmp_path)


def _run_probe(module_name="", failure="", model_dir=""):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(repo_root),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        MODELSCOPE_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )
    env.pop("FUNASR_STRICT_IMPORT", None)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, module_name, failure, model_dir],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("failure", ["import", "runtime"])
@pytest.mark.parametrize(
    "module_name",
    ["funasr.models.campplus.cluster_backend", "funasr.models.campplus.utils"],
)
def test_unavailable_speaker_dependencies_preserve_cause(module_name, failure, speaker_config):
    _run_probe(module_name, failure, model_dir=speaker_config)


def test_available_speaker_dependencies_construct_backend(speaker_config):
    _run_probe(model_dir=speaker_config)
