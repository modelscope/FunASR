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
