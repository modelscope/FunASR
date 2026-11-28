"""Offline regressions for training and validation JSONL shard selection."""

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch, name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["audio_datasets", "kws_datasets"])
def dataset_class(monkeypatch, request):
    # Load only the dataset and index, without FunASR's optional audio dependencies.
    _load_module(monkeypatch, "funasr.register", "funasr/register.py")
    monkeypatch.setitem(sys.modules, "librosa", types.ModuleType("librosa"))
    load_utils = types.ModuleType("funasr.utils.load_utils")

    def unexpected_audio_load(*args, **kwargs):
        pytest.fail("Index construction must not load audio")

    load_utils.extract_fbank = unexpected_audio_load
    load_utils.load_audio_text_image_video = unexpected_audio_load
    monkeypatch.setitem(sys.modules, "funasr.utils.load_utils", load_utils)
    _load_module(
        monkeypatch,
        "dataset_index_regression",
        "funasr/datasets/audio_datasets/index_ds.py",
    )
    module = _load_module(
        monkeypatch,
        "dataset_regression",
        f"funasr/datasets/{request.param}/datasets.py",
    )
    return module.AudioDataset if request.param == "audio_datasets" else module.KwsMTDataset


@pytest.mark.parametrize("is_training", [False, True, None])
@pytest.mark.parametrize("data_split_i", [0, 1])
def test_dataset_index_shard_selection(tmp_path, dataset_class, is_training, data_split_i):
    shards = []
    for i in range(4):
        shard = tmp_path / f"shard_{i}.jsonl"
        shard.write_text(json.dumps({"text": f"sample_{i}"}) + "\n", encoding="utf-8")
        shards.append(str(shard))
    shard_list = tmp_path / "shards.list"
    shard_list.write_text("\n".join(shards) + "\n", encoding="utf-8")

    # None exercises the default training mode with no explicit flag.
    kwargs = {} if is_training is None else {"is_training": is_training}
    dataset = dataset_class(
        str(shard_list),
        index_ds="IndexDSJsonl",
        data_split_num=2,
        data_split_i=data_split_i,
        **kwargs,
    )

    expected = [f"sample_{i}" for i in range(4)]
    if is_training is not False:
        expected = expected[data_split_i * 2 : (data_split_i + 1) * 2]
    assert len(dataset) == len(expected)
    assert [dataset.index_ds[i] for i in range(len(dataset))] == expected
