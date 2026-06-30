import importlib.util
import sys
import types
from pathlib import Path


def load_generate_subtitle_module():
    script_path = Path(__file__).resolve().parents[1] / "examples" / "subtitle" / "generate_subtitle.py"
    spec = importlib.util.spec_from_file_location("generate_subtitle", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_subtitle_requests_sentence_timestamps_and_writes_segmented_srt(
    monkeypatch, tmp_path
):
    module = load_generate_subtitle_module()
    captured = {}

    class FakeAutoModel:
        def __init__(self, **kwargs):
            captured["model_kwargs"] = kwargs

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            if kwargs.get("sentence_timestamp") and kwargs.get("output_timestamp"):
                return [
                    {
                        "text": "<|zh|>First segment Second segment",
                        "sentence_info": [
                            {"start": 0, "end": 3500, "text": "<|zh|>First segment"},
                            {"start": 3500, "end": 7200, "sentence": "Second segment", "spk": 1},
                        ],
                    }
                ]
            return [{"text": "<|zh|>First segment Second segment"}]

    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake media")
    output_path = tmp_path / "sub.srt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_subtitle.py",
            str(input_path),
            "-o",
            str(output_path),
            "--lang",
            "zh",
            "--device",
            "cpu",
            "--spk",
        ],
    )

    module.main()

    assert captured["model_kwargs"]["punc_model"] == "ct-punc"
    assert captured["generate_kwargs"]["language"] == "zh"
    assert captured["generate_kwargs"]["sentence_timestamp"] is True
    assert captured["generate_kwargs"]["output_timestamp"] is True
    assert captured["generate_kwargs"]["return_time_stamps"] is True
    assert output_path.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,000 --> 00:00:03,500\n"
        "First segment\n\n"
        "2\n"
        "00:00:03,500 --> 00:00:07,200\n"
        "[Speaker 1] Second segment\n\n"
    )


def test_generate_subtitle_falls_back_to_timestamp_bounds_when_sentence_info_missing(
    monkeypatch, tmp_path
):
    module = load_generate_subtitle_module()

    class FakeAutoModel:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            return [
                {
                    "text": "<|zh|>Full text",
                    "timestamp": [[250, 1000], [1000, 3200]],
                }
            ]

    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)

    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake media")
    output_path = tmp_path / "sub.srt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_subtitle.py", str(input_path), "-o", str(output_path), "--device", "cpu"],
    )

    module.main()

    assert output_path.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,250 --> 00:00:03,200\n"
        "Full text\n\n"
    )
