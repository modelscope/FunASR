"""Offline model options of runtime/python/websocket/funasr_wss_server.py.

The README documents ``--asr_model`` for this server. The server parsed ``--asr_model`` and
``--asr_model_revision`` but built its offline model from a fixed "paraformer-zh" v2.0.4,
so both options had no effect.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "runtime/python/websocket/funasr_wss_server.py"


def load_server(monkeypatch, *options):
    """Run the server module with these command-line options; it builds its models on import."""
    pytest.importorskip("numpy")
    # Each AutoModel call returns its keyword arguments.
    funasr = types.ModuleType("funasr")
    funasr.AutoModel = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, "funasr", funasr)
    # Stand-ins for the modules the server uses to serve requests, not to build its models.
    stand_ins = {
        name: types.ModuleType(name)
        for name in ("websockets", "torch", "scipy", "scipy.spatial", "scipy.spatial.distance")
    }
    stand_ins["scipy.spatial.distance"].cosine = None
    for name, module in stand_ins.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", [str(SERVER_PATH), *options])
    spec = importlib.util.spec_from_file_location("funasr_wss_server_asr_model", SERVER_PATH)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    server.EXECUTOR.shutdown(wait=True)
    return server


def test_default_offline_model_is_unchanged(monkeypatch):
    server = load_server(monkeypatch)
    assert server.model_asr.model == "paraformer-zh"
    assert server.model_asr.model_revision == "v2.0.4"


@pytest.mark.parametrize(
    "options, model, revision",
    [
        (["--asr_model", "paraformer-en"], "paraformer-en", "v2.0.4"),
        (
            ["--asr_model", "/models/my-paraformer", "--asr_model_revision", "master"],
            "/models/my-paraformer",
            "master",
        ),
    ],
)
def test_asr_model_options_select_the_offline_model(monkeypatch, options, model, revision):
    server = load_server(monkeypatch, *options)
    assert server.model_asr.model == model
    assert server.model_asr.model_revision == revision
    # The streaming model keeps its own options.
    assert server.model_asr_streaming.model == server.args.asr_model_online
