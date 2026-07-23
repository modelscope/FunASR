import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = REPO_ROOT / "funasr" / "bin" / "_server_app.py"


def load_server_app(monkeypatch):
    class DummyFastAPI:
        def __init__(self, *args, **kwargs):
            self.state = types.SimpleNamespace()
            self.routes = {}

        def post(self, path, *args, **kwargs):
            def decorator(func):
                self.routes[("POST", path)] = func
                return func

            return decorator

        def get(self, path, *args, **kwargs):
            def decorator(func):
                self.routes[("GET", path)] = func
                return func

            return decorator

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


def install_dummy_funasr(monkeypatch, fail_once_for_models=()):
    remaining_failures = {model: 1 for model in fail_once_for_models}

    class DummyAutoModel:
        instances = []
        attempts = []

        def __init__(self, **kwargs):
            self.__class__.attempts.append(kwargs)
            model = kwargs.get("model")
            if remaining_failures.get(model, 0):
                remaining_failures[model] -= 1
                raise RuntimeError(f"{model} unavailable")
            self.kwargs = kwargs
            self.__class__.instances.append(kwargs)

        def generate(self, **kwargs):
            if self.kwargs.get("model") == "fsmn-vad":
                return [{"value": [[0, 1000]]}]
            return [{"text": "transcript"}]

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
            return cls()

        def generate(self, inputs, **kwargs):
            return [{"text": "transcript"} for _ in inputs]

    monkeypatch.setitem(sys.modules, "funasr.models", types.ModuleType("funasr.models"))
    monkeypatch.setitem(
        sys.modules, "funasr.models.fun_asr_nano", types.ModuleType("funasr.models.fun_asr_nano")
    )
    vllm_module = types.ModuleType("funasr.models.fun_asr_nano.inference_vllm")
    vllm_module.FunASRNanoVLLM = DummyVLLM
    monkeypatch.setitem(sys.modules, "funasr.models.fun_asr_nano.inference_vllm", vllm_module)
    return DummyVLLM


class DummyUpload:
    filename = "audio.wav"

    async def read(self):
        return b"not-a-real-wave-file"


def transcribe_nano(app):
    transcribe = app.routes[("POST", "/v1/audio/transcriptions")]
    return asyncio.run(
        transcribe(
            file=DummyUpload(),
            model="fun-asr-nano",
            language=None,
            response_format="json",
            spk=False,
        )
    )


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


def test_extract_language_from_sensevoice_text(monkeypatch):
    module = load_server_app(monkeypatch)

    assert module.extract_language_from_asr_text("<|en|><|NEUTRAL|><|Speech|>hello") == "en"
    assert module.extract_language_from_asr_text("<|yue|> nei hou") == "yue"
    assert module.extract_language_from_asr_text("plain transcript") is None


def test_resolve_transcription_language_prefers_request_then_detection(monkeypatch):
    module = load_server_app(monkeypatch)

    assert module.resolve_transcription_language("ja", {"language": "en"}) == "ja"
    assert module.resolve_transcription_language("auto", {"language": "en"}) == "en"
    assert module.resolve_transcription_language(None, {"language": "ko"}) == "ko"
    assert module.resolve_transcription_language(None, {}) == "unknown"


def test_resolve_transcription_language_does_not_default_to_chinese(monkeypatch):
    module = load_server_app(monkeypatch)

    assert module.resolve_transcription_language(None, {}) != "zh"


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


def test_fun_asr_nano_reuses_fallback_after_vllm_failure(monkeypatch):
    module = load_server_app(monkeypatch)
    DummyAutoModel = install_dummy_funasr(monkeypatch)
    DummyVLLM = install_dummy_vllm(monkeypatch, raise_on_load=True)
    monkeypatch.setattr(module.sf, "info", lambda path: types.SimpleNamespace(duration=1.0))

    app = module.create_app(device="cuda", preload_model="fun-asr-nano", hub="ms")

    for _ in range(2):
        assert transcribe_nano(app) == {"text": "transcript"}

    nano_fallbacks = [
        config
        for config in DummyAutoModel.instances
        if config.get("model") == "FunAudioLLM/Fun-ASR-Nano-2512"
    ]
    assert len(DummyVLLM.calls) == 1
    assert len(nano_fallbacks) == 1


def test_partial_vllm_setup_is_not_cached_after_fallback_failure(monkeypatch):
    module = load_server_app(monkeypatch)
    nano_model = "FunAudioLLM/Fun-ASR-Nano-2512"
    DummyAutoModel = install_dummy_funasr(
        monkeypatch,
        fail_once_for_models=("fsmn-vad", nano_model),
    )
    DummyVLLM = install_dummy_vllm(monkeypatch)
    monkeypatch.setattr(
        module.sf,
        "read",
        lambda stream: (module.np.ones(16000, dtype=module.np.float32), 16000),
    )

    app = module.create_app(device="cuda", preload_model="sensevoice", hub="ms")

    with pytest.raises(RuntimeError, match=f"{nano_model} unavailable"):
        transcribe_nano(app)

    assert app.state.engine is None

    assert transcribe_nano(app) == {"text": "transcript"}
    assert len(DummyVLLM.calls) == 2
    assert len([attempt for attempt in DummyAutoModel.attempts if attempt.get("model") == "fsmn-vad"]) == 2


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
