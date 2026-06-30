import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = (
    REPO_ROOT
    / "examples"
    / "industrial_data_pretraining"
    / "fun_asr_nano"
    / "serve_vllm.py"
)


def load_service_module(monkeypatch):
    fastapi_stub = types.ModuleType("fastapi")

    class FastAPIStub:
        def __init__(self, *args, **kwargs):
            pass

        def on_event(self, *args, **kwargs):
            return lambda func: func

        def post(self, *args, **kwargs):
            return lambda func: func

        def websocket(self, *args, **kwargs):
            return lambda func: func

    fastapi_stub.FastAPI = FastAPIStub
    fastapi_stub.File = lambda *args, **kwargs: None
    fastapi_stub.Form = lambda *args, **kwargs: None
    fastapi_stub.UploadFile = object
    fastapi_stub.WebSocket = object
    fastapi_stub.WebSocketDisconnect = Exception

    responses_stub = types.ModuleType("fastapi.responses")

    class JSONResponseStub:
        def __init__(self, content=None):
            self.content = content

    responses_stub.JSONResponse = JSONResponseStub

    funasr_stub = types.ModuleType("funasr")
    funasr_stub.AutoModel = object

    nano_stub = types.ModuleType("funasr.models.fun_asr_nano.inference_vllm")
    nano_stub.FunASRNanoVLLM = object

    vad_stub = types.ModuleType("funasr.models.fsmn_vad_streaming.dynamic_vad")
    vad_stub.DynamicStreamingVAD = object

    monkeypatch.setitem(sys.modules, "fastapi", fastapi_stub)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_stub)
    monkeypatch.setitem(sys.modules, "uvicorn", types.ModuleType("uvicorn"))
    monkeypatch.setitem(sys.modules, "soundfile", types.ModuleType("soundfile"))
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "funasr", funasr_stub)
    monkeypatch.setitem(sys.modules, "funasr.models", types.ModuleType("funasr.models"))
    monkeypatch.setitem(
        sys.modules,
        "funasr.models.fun_asr_nano",
        types.ModuleType(
        "funasr.models.fun_asr_nano"
        ),
    )
    monkeypatch.setitem(
        sys.modules, "funasr.models.fun_asr_nano.inference_vllm", nano_stub
    )
    monkeypatch.setitem(
        sys.modules,
        "funasr.models.fsmn_vad_streaming",
        types.ModuleType("funasr.models.fsmn_vad_streaming"),
    )
    monkeypatch.setitem(
        sys.modules, "funasr.models.fsmn_vad_streaming.dynamic_vad", vad_stub
    )

    module_name = "serve_vllm_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SERVICE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_openai_verbose_json_keeps_segment_speaker(monkeypatch):
    module = load_service_module(monkeypatch)

    response = module.build_openai_verbose_json(
        {
            "duration": 1.2,
            "text": "hello",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "text": "hello",
                    "words": [{"word": "hello", "start": 0.0, "end": 1.2}],
                    "speaker": "SPK0",
                }
            ],
        },
        language="en",
    )

    assert response["segments"][0]["speaker"] == "SPK0"
