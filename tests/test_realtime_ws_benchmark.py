import asyncio
import importlib.util
import json
import sys
from pathlib import Path


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
