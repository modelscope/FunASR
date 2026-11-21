"""Client sample rate in runtime/python/websocket/funasr_wss_server.py.

The websocket protocol (runtime/docs/websocket_protocol.md) sends ``audio_fs``
with PCM audio, and its real-time recognition takes 8000 and 16000 Hz.
funasr_wss_client.py sends the sample rate of its WAV file or its --audio_fs,
and the HTML5 client the sample rate of its WAV file. The C++ servers resample
that audio to the model's rate. The Python server used ``audio_fs`` only to time
the frames, and gave AutoModel the PCM frames without it, so its models read
8 kHz audio as 16 kHz audio. AutoModel reads PCM input at the ``fs`` it is given
(16000 by default) and resamples it to the model's rate.
"""

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "runtime/python/websocket/funasr_wss_server.py"


class FakeWebSocket:
    def __init__(self, incoming):
        self.incoming = iter(incoming)
        self.sent = []
        self.path = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.incoming)
        except StopIteration:
            raise StopAsyncIteration

    async def send(self, message):
        self.sent.append(message)

    async def close(self):
        pass


@pytest.fixture
def server(monkeypatch):
    for name in ("websockets", "scipy", "torch"):
        pytest.importorskip(name)
    # The server builds its models with AutoModel when it is imported.
    fake_funasr = types.ModuleType("funasr")
    fake_funasr.AutoModel = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)
    monkeypatch.setattr(sys, "argv", [str(SERVER_PATH)])
    spec = importlib.util.spec_from_file_location("funasr_wss_server_audio_fs", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    module.EXECUTOR.shutdown(wait=True)


def _model_rates(server, audio_fs):
    """Stream 1.2 s of audio in 2pass mode and return the rate each model reads it at."""
    rates = {"vad": [], "online": [], "offline": [], "speaker": []}

    def model(name, result):
        def generate(input, **kwargs):
            # AutoModel reads PCM input at 16000 Hz unless it is given fs.
            rates[name].append(kwargs.get("fs", 16000))
            return [result]

        return SimpleNamespace(generate=generate)

    embedding = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: None))
    server.model_vad = model("vad", {"value": []})
    server.model_asr_streaming = model("online", {"text": "online"})
    server.model_asr = model("offline", {"text": "offline"})
    server.model_sv = model("speaker", {"spk_embedding": [embedding]})
    server.model_punc = None
    config = {
        "mode": "2pass",
        "chunk_size": [5, 10, 5],
        "chunk_interval": 10,
        "wav_name": "demo",
        "is_speaking": True,
    }
    if audio_fs is not None:
        config["audio_fs"] = audio_fs
    # 60 ms of PCM16 per frame, as funasr_wss_client.py sends with these options.
    frame_rate = audio_fs if isinstance(audio_fs, int) and audio_fs > 0 else 16000
    frame = b"\x01\x00" * (frame_rate * 60 // 1000)
    websocket = FakeWebSocket(
        [
            json.dumps(config),
            *[frame] * 20,
            json.dumps({"is_speaking": False, "is_end": True}),
        ]
    )

    asyncio.run(server.ws_serve(websocket))

    assert json.loads(websocket.sent[-1])["is_final"] is True
    return rates


@pytest.mark.parametrize(
    "audio_fs, expected",
    [
        # funasr_wss_client.py --audio_in with an 8 kHz WAV file, or --audio_fs 8000
        (8000, 8000),
        (16000, 16000),
        # The microphone mode of funasr_wss_client.py sends 16 kHz audio without audio_fs.
        (None, 16000),
    ],
)
def test_models_read_pcm_at_the_client_sample_rate(server, audio_fs, expected):
    rates = _model_rates(server, audio_fs)
    # VAD and online ASR read every frame; offline ASR and the speaker model the whole input.
    assert all(rates.values())
    assert {name: set(calls) for name, calls in rates.items()} == {
        name: {expected} for name in rates
    }


@pytest.mark.parametrize("audio_fs", [44100, 48000, 22050])
def test_wav_sample_rates_reach_the_models(server, audio_fs):
    # The bundled clients send the rate of the WAV file they stream.
    rates = _model_rates(server, audio_fs)
    assert {name: set(calls) for name, calls in rates.items()} == {
        name: {audio_fs} for name in rates
    }


@pytest.mark.parametrize("audio_fs", [0, -8000, 44101, "8k"])
def test_unsupported_audio_fs_keeps_the_models_at_16000(server, audio_fs):
    # 0 or a negative rate would make every model call fail, and a rate such as
    # 44101 would make the resampler build a very large kernel on every call.
    rates = _model_rates(server, audio_fs)
    assert all(rates.values())
    assert {name: set(calls) for name, calls in rates.items()} == {name: {16000} for name in rates}


def test_every_model_call_names_its_sample_rate(server):
    # A connection without audio_fs still passes fs, so it never reads its audio at
    # another connection's rate through the shared models' settings.
    seen = []

    def model(result):
        def generate(input, **kwargs):
            seen.append("fs" in kwargs)
            return [result]

        return SimpleNamespace(generate=generate)

    embedding = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: None))
    server.model_vad = model({"value": []})
    server.model_asr_streaming = model({"text": "online"})
    server.model_asr = model({"text": "offline"})
    server.model_sv = model({"spk_embedding": [embedding]})
    server.model_punc = None
    config = {
        "mode": "2pass",
        "chunk_size": [5, 10, 5],
        "chunk_interval": 10,
        "wav_name": "demo",
        "is_speaking": True,
    }
    frame = b"\x01\x00" * 960
    websocket = FakeWebSocket(
        [json.dumps(config), *[frame] * 20, json.dumps({"is_speaking": False, "is_end": True})]
    )
    asyncio.run(server.ws_serve(websocket))
    assert seen and all(seen)
