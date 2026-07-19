import io
import sys
import types
from contextlib import redirect_stdout
from unittest.mock import patch

from funasr import cli


class DummyAutoModel:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [{"text": "hello"}]


class SubtitleAutoModel(DummyAutoModel):
    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        if kwargs.get("sentence_timestamp") and kwargs.get("output_timestamp"):
            return [
                {
                    "text": "<|zh|>第一句。第二句。",
                    "sentence_info": [
                        {"start": 0, "end": 1200, "text": "<|zh|>第一句。"},
                        {"start": 1200, "end": 2600, "sentence": "第二句。"},
                    ],
                }
            ]
        return [{"text": "<|zh|>第一句。第二句。"}]


def test_cli_passes_hub_to_auto_model(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not a real wav")
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )

    DummyAutoModel.instances = []
    argv = ["funasr", "--hub", "hf", str(audio_path)]

    with (
        patch.object(sys, "argv", argv),
        patch.dict(sys.modules, {"torch": fake_torch}),
        patch("funasr.AutoModel", DummyAutoModel),
        redirect_stdout(io.StringIO()) as stdout,
    ):
        cli.main()

    assert DummyAutoModel.instances[0].kwargs["hub"] == "hf"
    assert stdout.getvalue().strip() == "hello"


def test_cli_routes_multiple_hotwords_to_paraformer_hotword(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not a real wav")
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )

    DummyAutoModel.instances = []
    argv = [
        "funasr",
        "--model",
        "paraformer",
        "--hotwords",
        "FunASR, ModelScope",
        str(audio_path),
    ]

    with (
        patch.object(sys, "argv", argv),
        patch.dict(sys.modules, {"torch": fake_torch}),
        patch("funasr.AutoModel", DummyAutoModel),
        redirect_stdout(io.StringIO()),
    ):
        cli.main()

    generate_kwargs = DummyAutoModel.instances[0].generate_kwargs
    assert generate_kwargs["hotword"] == "FunASR ModelScope"
    assert "hotwords" not in generate_kwargs


def test_timestamp_bounds_skip_malformed_entries():
    assert cli._timestamp_bounds_ms(
        {
            "timestamp": [
                [None, 1000],
                ["bad", "2000"],
                ["1200.0", "2600.0"],
                {"start": "0.5", "end": "1.0"},
                {"start_time": "oops", "end_time": "2.0"},
            ]
        }
    ) == (500, 2600)


def test_cli_srt_requests_sentence_timestamps_and_writes_segmented_output(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not a real wav")
    out_dir = tmp_path / "subs"
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )

    SubtitleAutoModel.instances = []
    argv = [
        "funasr",
        str(audio_path),
        "--output-format",
        "srt",
        "--output-dir",
        str(out_dir),
        "--lang",
        "zh",
    ]

    with (
        patch.object(sys, "argv", argv),
        patch.dict(sys.modules, {"torch": fake_torch}),
        patch("funasr.AutoModel", SubtitleAutoModel),
        redirect_stdout(io.StringIO()),
    ):
        cli.main()

    instance = SubtitleAutoModel.instances[0]
    assert instance.kwargs["punc_model"] == "ct-punc"
    assert instance.generate_kwargs["language"] == "zh"
    assert instance.generate_kwargs["sentence_timestamp"] is True
    assert instance.generate_kwargs["output_timestamp"] is True
    assert instance.generate_kwargs["return_time_stamps"] is True
    assert (out_dir / "sample.srt").read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,200\n"
        "第一句。\n\n"
        "2\n"
        "00:00:01,200 --> 00:00:02,600\n"
        "第二句。\n"
    )
