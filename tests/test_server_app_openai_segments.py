import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = REPO_ROOT / "funasr" / "bin" / "_server_app.py"
SERVER_CLI_PATH = REPO_ROOT / "funasr" / "bin" / "server.py"


def load_server_app(monkeypatch):
    class DummyFastAPI:
        def __init__(self, *args, **kwargs):
            self.state = types.SimpleNamespace()
            self.routes = {}
            self.metadata = kwargs
            self.middleware = []

        def add_middleware(self, middleware_class, **kwargs):
            self.middleware.append((middleware_class, kwargs))

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
    fastapi_stub.__path__ = []

    middleware_stub = types.ModuleType("fastapi.middleware")
    middleware_stub.__path__ = []

    cors_stub = types.ModuleType("fastapi.middleware.cors")

    class DummyCORSMiddleware:
        pass

    cors_stub.CORSMiddleware = DummyCORSMiddleware

    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.JSONResponse = lambda content=None: content

    monkeypatch.setitem(sys.modules, "fastapi", fastapi_stub)
    monkeypatch.setitem(sys.modules, "fastapi.middleware", middleware_stub)
    monkeypatch.setitem(sys.modules, "fastapi.middleware.cors", cors_stub)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_stub)

    module_name = "funasr_server_app_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SERVICE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_server_cli():
    module_name = "funasr_server_cli_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SERVER_CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def install_dummy_funasr(
    monkeypatch,
    fail_once_for_models=(),
    generated_text="transcript",
    generated_result=None,
):
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
            if generated_result is not None:
                return [generated_result.copy()]
            return [{"text": generated_text}]

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


def test_verbose_json_reports_sensevoice_detected_language(monkeypatch):
    module = load_server_app(monkeypatch)
    install_dummy_funasr(
        monkeypatch,
        generated_text="<|en|><|NEUTRAL|><|Speech|>hello",
    )
    monkeypatch.setattr(module.sf, "info", lambda path: types.SimpleNamespace(duration=1.25))
    app = module.create_app(device="cpu", preload_model="sensevoice")
    transcribe = app.routes[("POST", "/v1/audio/transcriptions")]

    response = asyncio.run(
        transcribe(
            file=DummyUpload(),
            model="sensevoice",
            language=None,
            response_format="verbose_json",
            spk=False,
        )
    )

    assert response["language"] == "en"
    assert response["text"] == "hello"
    assert response["segments"] == [
        {"id": 0, "start": 0.0, "end": 1.25, "text": "hello", "words": []}
    ]


def test_openai_verbose_json_preserves_speaker_labels(monkeypatch):
    module = load_server_app(monkeypatch)

    response = module.build_openai_verbose_json(
        {
            "language": "zh",
            "duration": 1.25,
            "text": "hello",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.25,
                    "text": "hello",
                    "speaker": "SPK0",
                }
            ],
        },
        requested_language=None,
    )

    assert response["segments"] == [
        {
            "id": 0,
            "start": 0.0,
            "end": 1.25,
            "text": "hello",
            "words": [],
            "speaker": "SPK0",
        }
    ]


def test_attach_speaker_labels_runs_diarization_pipeline(monkeypatch):
    import torch

    module = load_server_app(monkeypatch)
    calls = []

    class DummyClusterBackend:
        def __init__(self, merge_thr):
            assert merge_thr == 0.78

        def to(self, device):
            assert device == "cpu"
            return self

        def __call__(self, embeddings, oracle_num=None):
            assert embeddings.shape[1] == 2
            assert oracle_num is None
            calls.append("cluster")
            return module.np.zeros(embeddings.shape[0], dtype=int)

    class DummySpeakerModel:
        def generate(self, input, cache, is_final):
            assert input
            assert cache == {}
            assert is_final is True
            return [
                {"spk_embedding": torch.tensor([[1.0, 0.0]])}
                for _ in input
            ]

    def fake_sv_chunk(diarization_inputs, fs):
        assert fs == 16000
        assert diarization_inputs[0][:2] == [0.0, 2.0]
        calls.append("chunk")
        return [[0.0, 2.0, diarization_inputs[0][2]]]

    def fake_postprocess(chunks, vad_segments, labels, embeddings):
        assert vad_segments is None
        assert labels.tolist() == [0]
        assert embeddings.shape == (1, 2)
        calls.append("postprocess")
        return [[chunks[0][0], chunks[0][1], 0]]

    def fake_distribute_spk(sentences, speaker_timeline):
        assert speaker_timeline == [[0.0, 2.0, 0]]
        sentences[0]["spk"] = 0
        calls.append("distribute")

    funasr_stub = types.ModuleType("funasr")
    funasr_stub.__path__ = []
    models_stub = types.ModuleType("funasr.models")
    models_stub.__path__ = []
    campplus_stub = types.ModuleType("funasr.models.campplus")
    campplus_stub.__path__ = []
    cluster_stub = types.ModuleType("funasr.models.campplus.cluster_backend")
    cluster_stub.ClusterBackend = DummyClusterBackend
    utils_stub = types.ModuleType("funasr.models.campplus.utils")
    utils_stub.sv_chunk = fake_sv_chunk
    utils_stub.postprocess = fake_postprocess
    utils_stub.distribute_spk = fake_distribute_spk
    monkeypatch.setitem(sys.modules, "funasr", funasr_stub)
    monkeypatch.setitem(sys.modules, "funasr.models", models_stub)
    monkeypatch.setitem(sys.modules, "funasr.models.campplus", campplus_stub)
    monkeypatch.setitem(
        sys.modules, "funasr.models.campplus.cluster_backend", cluster_stub
    )
    monkeypatch.setitem(sys.modules, "funasr.models.campplus.utils", utils_stub)
    segments = [{"start": 0.0, "end": 2.0, "text": "hello"}]

    result = module.attach_speaker_labels(
        module.np.ones(32000, dtype=module.np.float32),
        16000,
        segments,
        DummySpeakerModel(),
        "cpu",
    )

    assert result == [
        {"start": 0.0, "end": 2.0, "text": "hello", "speaker": "SPK0"}
    ]
    assert calls == ["chunk", "cluster", "postprocess", "distribute"]


def test_fallback_reads_sentence_info_sentence_field(monkeypatch):
    module = load_server_app(monkeypatch)
    install_dummy_funasr(
        monkeypatch,
        generated_result={
            "text": "hello",
            "sentence_info": [
                {"start": 0, "end": 1250, "sentence": "hello", "spk": 1}
            ],
        },
    )
    monkeypatch.setattr(module.sf, "info", lambda path: types.SimpleNamespace(duration=1.25))
    app = module.create_app(device="cpu", preload_model="sensevoice")
    transcribe = app.routes[("POST", "/v1/audio/transcriptions")]

    response = asyncio.run(
        transcribe(
            file=DummyUpload(),
            model="sensevoice",
            language=None,
            response_format="verbose_json",
            spk=False,
        )
    )

    assert response["segments"] == [
        {
            "id": 0,
            "start": 0.0,
            "end": 1.25,
            "text": "hello",
            "words": [],
            "speaker": 1,
        }
    ]


def test_spk_request_lazily_loads_and_reuses_speaker_model(monkeypatch):
    module = load_server_app(monkeypatch)
    DummyAutoModel = install_dummy_funasr(monkeypatch)
    monkeypatch.setattr(module.sf, "info", lambda path: types.SimpleNamespace(duration=1.25))
    monkeypatch.setattr(
        module.sf,
        "read",
        lambda path: (module.np.ones(20000, dtype=module.np.float32), 16000),
    )
    attach_calls = []

    def fake_attach(audio_data, sample_rate, segments, speaker_model, device):
        attach_calls.append((sample_rate, speaker_model.kwargs["model"], device))
        segments[0]["speaker"] = "SPK0"
        return segments

    monkeypatch.setattr(module, "attach_speaker_labels", fake_attach, raising=False)
    app = module.create_app(
        device="cpu",
        preload_model="sensevoice",
        spk_model="iic/speech_eres2netv2_sv_zh-cn_16k-common",
    )
    transcribe = app.routes[("POST", "/v1/audio/transcriptions")]

    assert not any(
        instance.get("model") == "iic/speech_eres2netv2_sv_zh-cn_16k-common"
        for instance in DummyAutoModel.instances
    )

    for _ in range(2):
        response = asyncio.run(
            transcribe(
                file=DummyUpload(),
                model="sensevoice",
                language=None,
                response_format="verbose_json",
                spk=True,
            )
        )
        assert response["segments"][0]["speaker"] == "SPK0"

    speaker_models = [
        instance
        for instance in DummyAutoModel.instances
        if instance.get("model") == "iic/speech_eres2netv2_sv_zh-cn_16k-common"
    ]
    assert len(speaker_models) == 1
    assert attach_calls == [
        (16000, "iic/speech_eres2netv2_sv_zh-cn_16k-common", "cpu"),
        (16000, "iic/speech_eres2netv2_sv_zh-cn_16k-common", "cpu"),
    ]


def test_moss_service_uses_pinned_joint_transcription_config(monkeypatch):
    module = load_server_app(monkeypatch)
    DummyAutoModel = install_dummy_funasr(monkeypatch)

    module.create_app(device="cuda:0", preload_model="moss-transcribe-diarize")

    config = DummyAutoModel.instances[-1]
    assert config["model"] == "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    assert config["model_revision"] == "e8681d68e7042738ffca8ac8212bc8fcb1131ab8"
    assert config["hub"] == "hf"
    assert config["backend"] == "hf"
    assert config["trust_remote_code"] is True
    assert "vad_model" not in config
    assert "spk_model" not in config


def test_moss_verbose_json_preserves_native_speaker_segments(monkeypatch):
    module = load_server_app(monkeypatch)
    DummyAutoModel = install_dummy_funasr(
        monkeypatch,
        generated_result={
            "text": "hello again",
            "sentence_info": [
                {"start": 100, "end": 900, "spk": "S01", "text": "hello"},
                {"start": 950, "end": 1700, "spk": "S02", "sentence": "again"},
            ],
        },
    )
    monkeypatch.setattr(module.sf, "info", lambda path: types.SimpleNamespace(duration=1.7))
    app = module.create_app(device="cuda:0", preload_model="moss-transcribe-diarize")
    transcribe = app.routes[("POST", "/v1/audio/transcriptions")]

    response = asyncio.run(
        transcribe(
            file=DummyUpload(),
            model="moss-transcribe-diarize",
            language=None,
            response_format="verbose_json",
            spk=True,
        )
    )

    assert [segment["speaker"] for segment in response["segments"]] == ["S01", "S02"]
    assert [segment["text"] for segment in response["segments"]] == ["hello", "again"]
    assert not any(config.get("model") == "cam++" for config in DummyAutoModel.instances)


def test_server_versions_follow_package_version(monkeypatch):
    expected = (REPO_ROOT / "funasr" / "version.txt").read_text().strip()
    module = load_server_app(monkeypatch)
    server_module = load_server_cli()
    install_dummy_funasr(monkeypatch)

    app = module.create_app(device="cpu", preload_model="sensevoice")

    assert app.metadata["version"] == expected
    assert hasattr(server_module, "server_version_label")
    assert server_module.server_version_label() == f"FunASR Server v{expected}"


def test_server_cli_collects_repeated_cors_origins():
    module = load_server_cli()

    args = module.build_parser().parse_args(
        [
            "--cors-origin",
            "http://localhost:3000",
            "--cors-origin",
            "http://127.0.0.1:3000",
        ]
    )

    assert args.cors_origin == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


def test_server_cli_accepts_speaker_model():
    module = load_server_cli()

    args = module.build_parser().parse_args(
        ["--spk-model", "iic/speech_eres2netv2_sv_zh-cn-16k-common"]
    )

    assert args.spk_model == "iic/speech_eres2netv2_sv_zh-cn-16k-common"


def test_server_cors_is_disabled_by_default(monkeypatch):
    module = load_server_app(monkeypatch)
    install_dummy_funasr(monkeypatch)

    app = module.create_app(device="cpu", preload_model="sensevoice")

    assert app.middleware == []


def test_server_configures_normalized_trusted_origins(monkeypatch):
    module = load_server_app(monkeypatch)
    install_dummy_funasr(monkeypatch)

    app = module.create_app(
        device="cpu",
        preload_model="sensevoice",
        cors_origins=[
            " http://localhost:3000 ",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            " ",
        ],
    )

    assert app.middleware == [
        (
            module.CORSMiddleware,
            {
                "allow_origins": [
                    "http://localhost:3000",
                    "http://127.0.0.1:3000",
                ],
                "allow_credentials": False,
                "allow_methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Authorization", "Content-Type"],
            },
        )
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
