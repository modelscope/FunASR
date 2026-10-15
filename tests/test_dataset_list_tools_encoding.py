"""``scp2jsonl`` / ``sensevoice2jsonl`` / ``jsonl2scp`` text encoding.

The training data loader reads jsonl lists as UTF-8, but the conversion tools
opened ``wav.scp``/``text.txt`` and their outputs with the platform's locale
encoding. On Windows a UTF-8 Chinese transcript then failed to decode (cp1252)
or was rewritten in GBK (cp936), which the loader could not read back.
"""

import builtins
import json

import numpy as np
import pytest
import soundfile

from funasr.datasets.audio_datasets import jsonl2scp, scp2jsonl, sensevoice2jsonl

TEXT = "欢迎大家来体验达摩院推出的语音识别模型"


@pytest.fixture
def gbk_locale(monkeypatch):
    real_open = builtins.open

    def gbk_default_open(file, mode="r", *args, **kwargs):
        # Simulate a Chinese Windows locale: text mode without an explicit encoding.
        if "b" not in mode and kwargs.get("encoding") is None and len(args) < 2:
            kwargs["encoding"] = "cp936"
            kwargs.setdefault("errors", "replace")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", gbk_default_open)


def _write_wav(path):
    soundfile.write(str(path), np.zeros(1600, dtype="float32"), 16000)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_scp2jsonl_reads_and_writes_utf8(tmp_path, gbk_locale):
    wav = tmp_path / "utt1.wav"
    _write_wav(wav)
    wav_scp = tmp_path / "wav.scp"
    wav_scp.write_text(f"utt1 {wav}\n", encoding="utf-8")
    text = tmp_path / "text.txt"
    text.write_text(f"utt1 {TEXT}\n", encoding="utf-8")
    out = tmp_path / "train.jsonl"

    scp2jsonl.gen_jsonl_from_wav_text_list(
        [str(wav_scp), str(text)], ("source", "target"), jsonl_file_out=str(out)
    )

    assert _read_jsonl(out) == [
        {
            "key": "utt1",
            "source": str(wav),
            "source_len": 10,
            "target": TEXT,
            "target_len": len(TEXT),
        }
    ]


def test_sensevoice2jsonl_reads_and_writes_utf8(tmp_path, gbk_locale):
    wav = tmp_path / "utt1.wav"
    _write_wav(wav)
    files = {
        "source": f"utt1 {wav}\n",
        "target": f"utt1 {TEXT}\n",
        "text_language": "utt1 <|zh|>\n",
        "emo_target": "utt1 <|NEUTRAL|>\n",
        "event_target": "utt1 <|Speech|>\n",
    }
    paths = []
    for name, content in files.items():
        path = tmp_path / f"{name}.txt"
        path.write_text(content, encoding="utf-8")
        paths.append(str(path))
    out = tmp_path / "train.jsonl"

    # All rich labels are given, so no model is loaded.
    sensevoice2jsonl.gen_jsonl_from_wav_text_list(paths, list(files), jsonl_file_out=str(out))

    [row] = _read_jsonl(out)
    assert row["target"] == TEXT
    assert row["text_language"] == "<|zh|>"


def test_jsonl2scp_writes_utf8(tmp_path, gbk_locale):
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(
        json.dumps({"key": "utt1", "source": "/data/utt1.wav", "target": TEXT}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    wav_scp = tmp_path / "wav.scp"
    text = tmp_path / "text.txt"

    jsonl2scp.gen_scp_from_jsonl(str(jsonl), ("source", "target"), str(wav_scp), str(text))

    assert wav_scp.read_text(encoding="utf-8") == "utt1\t/data/utt1.wav\n"
    assert text.read_text(encoding="utf-8") == f"utt1\t{TEXT}\n"
