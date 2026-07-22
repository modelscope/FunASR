import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = REPO_ROOT / "funasr" / "bin" / "_server_app.py"


def load_server_app(monkeypatch):
    class DummyFastAPI:
        def __init__(self, *args, **kwargs):
            self.state = types.SimpleNamespace()

        def post(self, *args, **kwargs):
            return lambda func: func

        def get(self, *args, **kwargs):
            return lambda func: func

    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.FastAPI = DummyFastAPI
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


def install_dummy_funasr(monkeypatch):
    class DummyAutoModel:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.instances.append(kwargs)

    funasr_stub = types.ModuleType("funasr")
    funasr_stub.AutoModel = DummyAutoModel
    monkeypatch.setitem(sys.modules, "funasr", funasr_stub)
    return DummyAutoModel


def install_dummy_vllm(monkeypatch, raise_on_load=False):
    class DummyVLLM:
        calls = []

        @classmethod
        def from_pretrained(cls, **kwargs):
            cls.calls.append(kwargs)
            if raise_on_load:
                raise RuntimeError("vllm unavailable")
            return object()

    monkeypatch.setitem(sys.modules, "funasr.models", types.ModuleType("funasr.models"))
    monkeypatch.setitem(
        sys.modules, "funasr.models.fun_asr_nano", types.ModuleType("funasr.models.fun_asr_nano")
    )
    vllm_module = types.ModuleType("funasr.models.fun_asr_nano.inference_vllm")
    vllm_module.FunASRNanoVLLM = DummyVLLM
    monkeypatch.setitem(sys.modules, "funasr.models.fun_asr_nano.inference_vllm", vllm_module)
    return DummyVLLM


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


def test_default_fun_asr_nano_uses_requested_modelscope_hub(monkeypatch):
    module = load_server_app(monkeypatch)
    DummyAutoModel = install_dummy_funasr(monkeypatch)
    DummyVLLM = install_dummy_vllm(monkeypatch)

    module.create_app(device="cuda", preload_model="fun-asr-nano", hub="ms")

    assert DummyVLLM.calls[0]["model"] == "FunAudioLLM/Fun-ASR-Nano-2512"
    assert DummyVLLM.calls[0]["hub"] == "ms"
    assert DummyAutoModel.instances[0]["model"] == "fsmn-vad"


def test_default_fun_asr_nano_fallback_uses_requested_modelscope_hub(monkeypatch):
    module = load_server_app(monkeypatch)
    DummyAutoModel = install_dummy_funasr(monkeypatch)
    install_dummy_vllm(monkeypatch, raise_on_load=True)

    module.create_app(device="cuda", preload_model="fun-asr-nano", hub="ms")

    fallback = DummyAutoModel.instances[-1]
    assert fallback["model"] == "FunAudioLLM/Fun-ASR-Nano-2512"
    assert fallback["hub"] == "ms"


def test_custom_model_path_fallback_uses_empty_config_and_requested_hub(monkeypatch):
    module = load_server_app(monkeypatch)
    DummyAutoModel = install_dummy_funasr(monkeypatch)

    app = module.create_app(
        device="cpu",
        preload_model="sensevoice",
        model_path="org/custom-sensevoice",
        hub="hf",
    )

    assert DummyAutoModel.instances[0]["model"] == "org/custom-sensevoice"
    assert DummyAutoModel.instances[0]["hub"] == "hf"
    assert app.state.fallback_models["custom"] is not None
