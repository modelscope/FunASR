"""Hotwords that runtime/python/websocket/funasr_wss_client.py sends in file mode.

The websocket protocol (runtime/docs/websocket_protocol.md) sends ``hotwords`` as a
JSON object string that maps each hotword to a weight. The microphone mode builds it
from the --hotword file ("阿里巴巴 20" per line); the file mode sent the first word of
each line joined with commas instead.
"""

import ast
import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

CLIENT_PATH = Path(__file__).resolve().parents[1] / "runtime/python/websocket/funasr_wss_client.py"


def _load_client_function(name, namespace):
    # Importing the client would parse sys.argv, so only the function is compiled.
    tree = ast.parse(CLIENT_PATH.read_text(encoding="utf-8"), filename=str(CLIENT_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(CLIENT_PATH), "exec"), namespace)
    return namespace[name]


class RecordingWebSocket:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    async def close(self):
        pass


def _file_mode_hotwords(tmp_path, hotword):
    """Send 0.1 s of audio with record_from_scp and return the config message's hotwords."""
    audio_path = tmp_path / "audio.pcm"
    audio_path.write_bytes(b"\x00\x00" * 1600)
    args = SimpleNamespace(
        audio_in=str(audio_path),
        hotword=hotword,
        audio_fs=16000,
        use_itn=1,
        chunk_size=[5, 10, 5],
        chunk_interval=10,
        encoder_chunk_look_back=4,
        decoder_chunk_look_back=0,
        mode="offline",
        send_without_sleep=True,
        result_timeout=1.0,
    )
    websocket = RecordingWebSocket()
    namespace = {
        "asyncio": asyncio,
        "json": json,
        "os": os,
        "time": time,
        "args": args,
        "websocket": websocket,
        "latency_first_audio_time": {},
        "latency_last_audio_time": {},
        "latency_first_text_printed": {},
    }
    record_from_scp = _load_client_function("record_from_scp", namespace)

    async def send_file():
        acknowledgement = asyncio.Queue()
        acknowledgement.put_nowait({"is_end": True, "is_final": True})
        await record_from_scp(0, 1, acknowledgement)

    asyncio.run(send_file())
    return json.loads(websocket.sent[0])["hotwords"]


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ("阿里巴巴 20\n通义实验室 30\n", '{"阿里巴巴": 20, "通义实验室": 30}'),
        ("hello world 10\n阿里巴巴\n通义实验室 high\n", '{"hello world": 10}'),
    ],
    ids=["two_hotwords", "phrase_and_lines_without_integer_weight"],
)
def test_file_mode_sends_hotword_file_as_json_object(tmp_path, lines, expected):
    hotword_file = tmp_path / "hotwords.txt"
    hotword_file.write_text(lines, encoding="utf-8")

    assert _file_mode_hotwords(tmp_path, str(hotword_file)) == expected


@pytest.mark.parametrize("hotword", ["", "阿里巴巴 通义实验室"], ids=["empty", "not_a_file"])
def test_file_mode_sends_other_hotword_arguments_unchanged(tmp_path, hotword):
    assert _file_mode_hotwords(tmp_path, hotword) == hotword
