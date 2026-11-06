"""Exercise the manual client CLI without a speech model or external service."""

import asyncio
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import wave

import pytest
from websockets.legacy.server import serve


ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "examples/industrial_data_pretraining/fun_asr_nano/client_test.py"


@pytest.mark.parametrize("import_mode", ["prepend", "append", "importlib"])
def test_manual_client_does_not_request_pytest_fixtures(tmp_path, import_mode):
    client = tmp_path / "client_test.py"
    shutil.copyfile(CLIENT, client)
    sentinel = tmp_path / "test_sentinel.py"
    sentinel.write_text("def test_sentinel():\n    pass\n", encoding="utf-8")
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "--import-mode=" + import_mode, str(client), str(sentinel)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 test collected" in result.stdout, result.stdout
    assert "test_sentinel.py::test_sentinel" in result.stdout


def test_manual_client_help_preserves_cli_arguments():
    result = subprocess.run(
        [sys.executable, str(CLIENT), "--help"], cwd=CLIENT.parent,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for option in ("--server", "--file", "--hotwords"):
        assert option in result.stdout


@pytest.mark.parametrize("hotwords", ["", "hello,world"])
@pytest.mark.parametrize("missing_final", [False, True])
def test_manual_client_checks_real_loopback_protocol(tmp_path, hotwords, missing_final):
    audio_path = tmp_path / "short.wav"
    pcm = struct.pack("<1600h", *(i % 300 for i in range(1600)))
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(pcm)

    async def run():
        sessions = []
        received_hotwords = []

        async def handler(ws, path=None):
            chunks = []
            async for message in ws:
                if message == "START":
                    chunks = []
                    await ws.send(json.dumps({"event": "started"}))
                elif isinstance(message, bytes):
                    chunks.append(message)
                elif message.startswith("HOTWORDS:"):
                    received_hotwords.append(message[9:])
                    await ws.send(json.dumps({"event": "hotwords_set", "hotwords": hotwords.split(",")}))
                elif message == "STOP":
                    sessions.append(b"".join(chunks))
                    if not missing_final:
                        await ws.send(json.dumps({"is_final": True, "sentences": [
                            {"text": "local protocol", "start": 0, "end": 100},
                        ]}))
                    await ws.send(json.dumps({"event": "stopped"}))
                else:
                    raise AssertionError("Unexpected client command: " + message)

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            command = [sys.executable, str(CLIENT), "--server", f"ws://127.0.0.1:{port}",
                       "--file", str(audio_path), "--hotwords", hotwords]
            process = await asyncio.create_subprocess_exec(
                *command, cwd=CLIENT.parent,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        return process.returncode, stdout.decode(), stderr.decode(), sessions, received_hotwords

    code, output, errors, sessions, received_hotwords = asyncio.run(run())
    assert code == (1 if missing_final else 0), output + errors
    assert ("RESULTS: 0 passed, 3 failed" if missing_final else "RESULTS: 3 passed, 0 failed") in output
    assert sessions == ([pcm, b"\0" * 1600, pcm] if missing_final else [pcm, b"\0" * 1600, pcm, pcm])
    assert received_hotwords == ([hotwords] if hotwords else [])
