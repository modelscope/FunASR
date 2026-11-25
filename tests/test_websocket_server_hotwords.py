"""Client hotwords in runtime/python/websocket/funasr_wss_server.py.

The websocket protocol (runtime/docs/websocket_protocol.md) sends ``hotwords``
as a JSON object string that maps each hotword to a weight, for example
'{"阿里巴巴":20,"通义实验室":30}'. The HTML5, Java, Android and C++ clients
and the microphone mode of funasr_wss_client.py all send this format. The
Python server passed the string to AutoModel unchanged, and AutoModel splits a
hotword string on whitespace, so the JSON text became ``<unk>`` hotwords.
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
    spec = importlib.util.spec_from_file_location("funasr_wss_server_hotwords", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    module.EXECUTOR.shutdown(wait=True)


def _offline_hotwords(server, hotwords):
    """Send one offline request and return the hotword AutoModel.generate receives."""
    asr_calls = []

    def asr_generate(input, **kwargs):
        asr_calls.append(kwargs)
        return [{"text": "阿里巴巴"}]

    def no_speaker_model(*args, **kwargs):
        raise RuntimeError("no speaker model in this test")

    server.model_asr = SimpleNamespace(generate=asr_generate)
    server.model_vad = SimpleNamespace(generate=lambda input, **kwargs: [{"value": []}])
    server.model_sv = SimpleNamespace(generate=no_speaker_model)
    server.model_punc = None
    config = {
        "mode": "offline",
        "chunk_size": [5, 10, 5],
        "chunk_interval": 10,
        "wav_name": "demo",
        "is_speaking": True,
        "hotwords": hotwords,
        "itn": True,
    }
    websocket = FakeWebSocket(
        [
            json.dumps(config, ensure_ascii=False),
            b"\x01\x00" * 1600,
            json.dumps({"is_speaking": False, "is_end": True}),
        ]
    )

    asyncio.run(server.ws_serve(websocket))

    assert json.loads(websocket.sent[-1])["is_final"] is True
    assert len(asr_calls) == 1
    return asr_calls[0]["hotword"]


@pytest.mark.parametrize(
    "hotwords, expected",
    [
        # HTML5, Java, Android and C++ clients (compact JSON)
        ('{"阿里巴巴":20,"通义实验室":30}', "阿里巴巴 通义实验室"),
        # funasr_wss_client.py microphone mode (json.dumps)
        ('{"阿里巴巴": 20, "通义实验室": 30}', "阿里巴巴 通义实验室"),
        # an empty object: no hotwords
        ("{}", ""),
    ],
)
def test_protocol_hotwords_reach_the_model_as_words(server, hotwords, expected):
    assert _offline_hotwords(server, hotwords) == expected


# A space-separated string is what AutoModel expects already, and
# funasr_wss_client.py sends "" when --hotword is not set.
@pytest.mark.parametrize("hotwords", ["阿里巴巴 通义实验室", ""])
def test_plain_hotword_strings_are_unchanged(server, hotwords):
    assert _offline_hotwords(server, hotwords) == hotwords
