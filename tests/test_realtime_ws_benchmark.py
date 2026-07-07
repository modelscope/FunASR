import asyncio
import importlib.util
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
