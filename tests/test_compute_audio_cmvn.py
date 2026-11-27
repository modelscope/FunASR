"""compute_audio_cmvn writes am.mvn next to cmvn_file, without model weights."""

import io
import os

import torch

from funasr.bin import compute_audio_cmvn
from funasr.frontends.wav_frontend import load_cmvn
from funasr.register import tables


class _FbankDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.feats = [torch.tensor([[1.0, 2.0], [3.0, 6.0]]), torch.tensor([[5.0, 10.0]])]

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        return self.feats[index]

    def collator(self, samples):
        return {"speech": torch.stack(samples)}


class _BatchSampler(dict):
    def __init__(self, dataset, is_training=False, **kwargs):
        super().__init__(batch_size=1)


def test_am_mvn_is_written_next_to_a_cmvn_file_without_directory(tmp_path, monkeypatch):
    monkeypatch.setitem(tables.dataset_classes, "CmvnTestDataset", _FbankDataset)
    monkeypatch.setitem(tables.batch_sampler_classes, "CmvnTestSampler", _BatchSampler)
    monkeypatch.chdir(tmp_path)

    opened = []
    real_open = open

    def recording_open(path, *args, **kwargs):
        opened.append(path)
        # Keep a regression from touching the filesystem root.
        if os.path.isabs(path) or path.startswith(("/", "\\")):
            return io.StringIO()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(compute_audio_cmvn, "open", recording_open, raising=False)

    compute_audio_cmvn.main(
        dataset="CmvnTestDataset",
        dataset_conf={"batch_sampler": "CmvnTestSampler"},
        train_data_set_list="unused.jsonl",
        cmvn_file="cmvn.json",
    )

    assert opened == ["cmvn.json", "am.mvn"]
    cmvn = load_cmvn(str(tmp_path / "am.mvn"))
    # mean = [3, 6], variance = [8/3, 32/3]
    torch.testing.assert_close(cmvn[0], torch.tensor([-3.0, -6.0]))
    torch.testing.assert_close(cmvn[1], torch.tensor([(3 / 8) ** 0.5, (3 / 32) ** 0.5]))
