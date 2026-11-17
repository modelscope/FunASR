from pathlib import Path
import runpy
import sys
import types
from unittest.mock import Mock

import pytest

pytest.importorskip("flask")
pytest.importorskip("numpy")
pytest.importorskip("scipy")

SERVICE_PATH = (
    Path(__file__).resolve().parents[1] / "runtime/python/websocket/funASR-Service.py"
)


@pytest.fixture
def service(monkeypatch, tmp_path):
    # Stub model dependencies, but execute the real entrypoint and Flask.run.
    modules = {
        name: types.ModuleType(name)
        for name in (
            "funasr",
            "modelscope",
            "modelscope.pipelines",
            "modelscope.utils",
            "modelscope.utils.constant",
            "openai",
            "torch",
            "torchaudio",
        )
    }
    modules["funasr"].AutoModel = Mock()
    modules["modelscope.pipelines"].pipeline = Mock()
    modules["modelscope.utils.constant"].Tasks = types.SimpleNamespace()
    modules["openai"].OpenAI = Mock()
    modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FLASK_RUN_FROM_CLI", raising=False)
    monkeypatch.setenv("FLASK_SKIP_DOTENV", "1")
    run_server = Mock()
    monkeypatch.setattr("werkzeug.serving.run_simple", run_server)

    def load():
        namespace = runpy.run_path(str(SERVICE_PATH), run_name="__main__")
        return namespace["app"], run_server

    return load


@pytest.mark.parametrize("debug_env", [None, "0", "1"])
def test_direct_start_disables_debugger_and_reloader(service, monkeypatch, debug_env):
    if debug_env is None:
        monkeypatch.delenv("FLASK_DEBUG", raising=False)
    else:
        monkeypatch.setenv("FLASK_DEBUG", debug_env)

    app, run_server = service()

    run_server.assert_called_once()
    args, options = run_server.call_args
    assert args == ("0.0.0.0", 10099, app)
    assert app.debug is False
    assert options["use_debugger"] is False
    assert options["use_reloader"] is False


def test_routes_still_work_with_debug_disabled(service):
    app, _ = service()
    client = app.test_client()

    for endpoint in ("/Register_Speaker", "/AsrCamWithIdentify"):
        response = client.post(endpoint)
        assert response.status_code == 400
        assert response.get_json() == {"error": "No audio file provided"}

    response = client.post(
        "/calculate_similarity", json={"emb1": [1.0, 0.0], "emb2": [1.0, 0.0]}
    )
    assert response.status_code == 200
    assert response.get_json() == {"status": "success", "result": pytest.approx(1.0)}


def test_unhandled_route_error_returns_generic_500(service):
    app, _ = service()

    response = app.test_client().post("/calculate_similarity", json={})

    assert response.status_code == 500
    assert b"Internal Server Error" in response.data
    assert b"Traceback" not in response.data
    assert b"Werkzeug Debugger" not in response.data
