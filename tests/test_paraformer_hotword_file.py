"""Hotword ``.txt`` parsing for SeACo-Paraformer and Contextual Paraformer.

``generate_hotwords_list`` turns every line of a hotword file into a list of
token ids. A blank line (for example an extra newline at the end of the file)
used to become an empty hotword, and the zero length then made
``pack_padded_sequence`` raise during decoding. The file was also read with the
platform's locale encoding, so a UTF-8 Chinese hotword file decoded into
``<unk>`` tokens (or raised) on Windows.
"""

import builtins
import os
import types

import pytest
import torch

from funasr.models.contextual_paraformer.model import ContextualParaformer
from funasr.models.seaco_paraformer.model import SeacoParaformer

SOS = 1
VOCAB = {"<unk>": 0, "<s>": SOS, "阿": 2, "里": 3, "巴": 4, "云": 5}


class _Tokenizer:
    def tokens2ids(self, tokens):
        return [VOCAB.get(t, VOCAB["<unk>"]) for t in tokens]


def _generate(model_cls, tmp_path, content):
    # Released Paraformer models ship a seg_dict next to am.mvn; it splits
    # Chinese words into characters.
    (tmp_path / "seg_dict").write_text("阿 阿\n里 里\n巴 巴\n云 云\n", encoding="utf-8")
    hotword_file = tmp_path / "hotwords.txt"
    hotword_file.write_text(content, encoding="utf-8")
    frontend = types.SimpleNamespace(cmvn_file=os.path.join(str(tmp_path), "am.mvn"))
    return model_cls.generate_hotwords_list(
        types.SimpleNamespace(sos=SOS),
        str(hotword_file),
        tokenizer=_Tokenizer(),
        frontend=frontend,
    )


@pytest.fixture(autouse=True)
def _fresh_seg_dict_cache():
    from funasr.models.seaco_paraformer import model as seaco_model

    seaco_model.load_seg_dict.cache_clear()
    yield
    seaco_model.load_seg_dict.cache_clear()


@pytest.mark.parametrize("model_cls", [SeacoParaformer, ContextualParaformer])
def test_blank_lines_in_hotword_file_are_skipped(model_cls, tmp_path):
    hotword_list = _generate(model_cls, tmp_path, "阿里巴巴\n\n   \n阿里云\n\n")

    assert hotword_list == [[2, 3, 4, 4], [2, 3, 5], [SOS]]
    # Every hotword must be packable by the bias encoder.
    lengths = torch.tensor([len(hw) for hw in hotword_list])
    torch.nn.utils.rnn.pack_padded_sequence(
        torch.zeros(len(hotword_list), int(lengths.max()), 4),
        lengths,
        batch_first=True,
        enforce_sorted=False,
    )


@pytest.mark.parametrize("model_cls", [SeacoParaformer, ContextualParaformer])
def test_hotword_file_is_read_as_utf8_regardless_of_locale(model_cls, tmp_path, monkeypatch):
    real_open = builtins.open

    def cp1252_default_open(file, mode="r", *args, **kwargs):
        # Simulate a Windows locale: text mode without an explicit encoding.
        if "b" not in mode and kwargs.get("encoding") is None and len(args) < 2:
            kwargs["encoding"] = "cp1252"
            kwargs.setdefault("errors", "replace")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", cp1252_default_open)

    hotword_list = _generate(model_cls, tmp_path, "阿里巴巴\n阿里云\n")

    assert hotword_list == [[2, 3, 4, 4], [2, 3, 5], [SOS]]
