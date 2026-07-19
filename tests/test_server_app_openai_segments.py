import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = REPO_ROOT / "funasr" / "bin" / "_server_app.py"


def load_server_app(monkeypatch):
    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.FastAPI = object
    fastapi_stub.UploadFile = object
    fastapi_stub.File = lambda *args, **kwargs: None
    fastapi_stub.Form = lambda *args, **kwargs: None
    fastapi_stub.HTTPException = Exception

    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.JSONResponse = lambda content=None: content

    monkeypatch.setitem(sys.modules, "fastapi", fastapi_stub)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_stub)

    module_name = "funasr_server_app_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SERVICE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fallback_segments_split_long_fun_asr_server_text(monkeypatch):
    module = load_server_app(monkeypatch)
    text = (
        "i believe that this nation should commit itself to achieving the goal before this decade is out "
        "of landing a man on the moon and returning him safely to the earth "
        "no single space project in this period will be more impressive to mankind "
        "or more important for the long range exploration of space"
    )

    segments = module.build_openai_fallback_segments(text, duration=21.0)

    assert len(segments) > 1
    assert segments[0]["start"] == 0.0
    assert segments[-1]["end"] == 21.0
    assert all(segment["end"] >= segment["start"] for segment in segments)
    assert all(len(segment["text"]) <= 80 for segment in segments)
    assert " ".join(segment["text"] for segment in segments) == text


def test_fallback_segments_keep_short_text_single_cue(monkeypatch):
    module = load_server_app(monkeypatch)

    assert module.build_openai_fallback_segments("hello", duration=1.25) == [
        {"start": 0.0, "end": 1.25, "text": "hello"}
    ]
