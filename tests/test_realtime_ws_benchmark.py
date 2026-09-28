import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = (
    REPO_ROOT
    / "examples"
    / "industrial_data_pretraining"
    / "fun_asr_nano"
    / "realtime_ws_benchmark.py"
)


def load_benchmark_module():
    module_name = "realtime_ws_benchmark_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_message_does_not_contribute_to_response_lag(monkeypatch):
    module = load_benchmark_module()
    messages = iter(
        [
            {"partial": "hello", "duration_ms": 1000},
            {"is_final": True, "sentences": [{"text": "hello"}], "duration_ms": 1200},
            {"event": "stopped"},
        ]
    )
    timestamps = iter([1.2, 2.0, 2.1])

    async def fake_receive_message(ws, timeout):
        return next(messages)

    monkeypatch.setattr(module, "receive_message", fake_receive_message)
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(timestamps))
    metrics = {
        "messages": 0,
        "result_messages": 0,
        "partial_messages": 0,
        "final_messages": 0,
        "events": {},
        "first_update_ms": None,
        "first_text_ms": None,
        "final_update_ms": None,
        "final_after_stop_ms": None,
        "response_lag_ms": [],
        "stopped": False,
        "errors": [],
    }

    asyncio.run(module.recv_results(object(), metrics, 0.0, {"value": 1.5}, 1.0))

    assert metrics["result_messages"] == 2
    assert metrics["partial_messages"] == 1
    assert metrics["final_messages"] == 1
    assert metrics["response_lag_ms"] == [200.0]
    assert metrics["final_update_ms"] == 2000.0
    assert metrics["final_after_stop_ms"] == 500.0


def test_client_ping_settings_are_forwarded_to_websocket_connect(monkeypatch):
    module = load_benchmark_module()
    args = module.parse_args(
        [
            "audio.wav",
            "--client-ping-interval",
            "7",
            "--client-ping-timeout",
            "11",
            "--no-pace",
        ]
    )
    connect_call = {}

    class FakeWebSocket:
        def __init__(self):
            self.messages = iter(
                [
                    {"event": "started"},
                    {"is_final": True, "sentences": [{"text": "hello"}]},
                    {"event": "stopped"},
                ]
            )

        async def send(self, _message):
            return None

        async def recv(self):
            return json.dumps(next(self.messages))

    class FakeConnection:
        async def __aenter__(self):
            return FakeWebSocket()

        async def __aexit__(self, *_args):
            return None

    def fake_connect(server, **kwargs):
        connect_call.update({"server": server, **kwargs})
        return FakeConnection()

    monkeypatch.setattr(module.websockets, "connect", fake_connect)

    result = asyncio.run(module.run_client(0, args, b"\0\0" * 1600, 0.1))

    assert result["errors"] == []
    assert result["client_ping_interval"] == 7.0
    assert result["client_ping_timeout"] == 11.0
    assert connect_call["ping_interval"] == 7.0
    assert connect_call["ping_timeout"] == 11.0


def test_client_ping_timeout_zero_disables_timeout():
    module = load_benchmark_module()

    args = module.parse_args(["audio.wav", "--client-ping-timeout", "0"])

    assert args.client_ping_timeout is None


@pytest.mark.parametrize(
    "messages,expected",
    [
        ([{"partial": ""}, {"partial": "hello"}, {"partial": "later"}], 1000.0),
        ([{"sentences": []}, {"sentences": [{"text": "hello"}]}], 1000.0),
        (
            [{"partial": ""}, {"is_final": True, "sentences": [{"text": "hello"}]}],
            1000.0,
        ),
        (
            [{"partial": " \t", "sentences": [{"text": "\n"}]}, {"partial": "hello"}],
            1000.0,
        ),
        ([{"partial": "hello"}, {"partial": "later"}], 0.0),
        ([{"sentences": []}, {"is_final": True, "sentences": []}], None),
        ([{"partial": None, "sentences": [{"text": None}]}], None),
    ],
)
def test_first_text_waits_for_nonblank_transcript(monkeypatch, messages, expected):
    module = load_benchmark_module()
    responses = iter([*messages, {"event": "stopped"}])
    timestamps = iter(float(i) for i in range(len(messages) + 1))

    async def receive_message(_ws, _timeout):
        return next(responses)

    monkeypatch.setattr(module, "receive_message", receive_message)
    monkeypatch.setattr(
        module, "time", SimpleNamespace(perf_counter=lambda: next(timestamps))
    )
    metrics = {
        "messages": 0,
        "result_messages": 0,
        "partial_messages": 0,
        "final_messages": 0,
        "events": {},
        "first_update_ms": None,
        "first_text_ms": None,
        "final_update_ms": None,
        "final_after_stop_ms": None,
        "response_lag_ms": [],
        "stopped": False,
        "errors": [],
    }
    asyncio.run(module.recv_results(object(), metrics, 0.0, {"value": None}, 1.0))

    assert metrics["first_update_ms"] == 0.0
    assert metrics["first_text_ms"] == expected
    assert metrics["stopped"] is True
    assert metrics["errors"] == []


@pytest.mark.parametrize(
    "latencies,expected_p50,expected_p95,count",
    [
        ([None, 0.0, 1000.0, 2000.0], 1000.0, 1900.0, 3),
        ([None, None], None, None, 0),
    ],
)
def test_first_text_summary_excludes_missing_text_but_counts_coverage(
    latencies, expected_p50, expected_p95, count
):
    module = load_benchmark_module()
    results = [
        {
            "audio_seconds": 1.0,
            "first_update_ms": 0.0,
            "first_text_ms": latency,
            "final_after_stop_ms": 0.0,
            "response_lag_ms_p95": 0.0,
            "partial_messages": 0,
            "final_messages": 1,
            "errors": [],
        }
        for latency in latencies
    ]
    summary = module.summarize(results, 3.0)

    assert summary["first_text_ms_p50"] == expected_p50
    assert summary["first_text_ms_p95"] == expected_p95
    assert summary["clients_with_text"] == count
    assert summary["clients"] == len(latencies)
    assert summary["first_update_ms_p50"] == 0.0


def test_first_text_round_trip_exports_client_and_summary_jsonl(tmp_path, capsys):
    module = load_benchmark_module()
    wav_path = tmp_path / "silence.wav"
    output_path = tmp_path / "metrics.jsonl"
    with wave.open(str(wav_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 3200)

    async def replay_server(ws, *_args):
        assert await ws.recv() == "START"
        await ws.send(json.dumps({"event": "started"}))
        assert isinstance(await ws.recv(), bytes)
        await ws.send(json.dumps({"sentences": [], "partial": ""}))
        assert isinstance(await ws.recv(), bytes)
        await ws.send(json.dumps({"partial": "hello"}))
        assert await ws.recv() == "STOP"
        await ws.send(json.dumps({"is_final": True, "sentences": [{"text": "hello"}]}))
        await ws.send(json.dumps({"event": "stopped"}))

    async def run():
        async with module.websockets.serve(replay_server, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            args = module.parse_args(
                [
                    str(wav_path),
                    "--server",
                    f"ws://127.0.0.1:{port}",
                    "--no-pace",
                    "--clients",
                    "2",
                    "--output-jsonl",
                    str(output_path),
                ]
            )
            return await asyncio.wait_for(module.async_main(args), timeout=5)

    assert asyncio.run(run()) == 0
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    clients = [row for row in records if row["type"] == "client"]
    summary = records[-1]
    assert len(clients) == 2
    for client in clients:
        assert client["errors"] == []
        assert 0 <= client["first_update_ms"] <= client["first_text_ms"]
        assert client["first_text_ms"] == round(client["first_text_ms"], 1)
        assert client["stopped"] is True
    assert summary["type"] == "summary"
    assert summary["clients_with_text"] == summary["clients"] == 2
    assert summary["first_text_ms_p50"] is not None
    assert summary["first_text_ms_p95"] is not None
    printed = capsys.readouterr().out
    assert "first text p50/p95 ms:" in printed
    assert "clients with text: 2/2" in printed
