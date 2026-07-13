import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = REPO_ROOT / "examples" / "industrial_data_pretraining" / "fun_asr_nano" / "serve_realtime_ws.py"


def load_service_module():
    websockets_stub = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(ConnectionClosed=Exception),
        serve=lambda *args, **kwargs: None,
    )
    sys.modules.setdefault("websockets", websockets_stub)

    module_name = "serve_realtime_ws_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SERVICE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_disable_speaker_and_bound_partial_window():
    module = load_service_module()

    assert hasattr(module, "build_arg_parser")
    args = module.build_arg_parser().parse_args([])

    assert args.enable_spk is False
    assert args.partial_window_sec == 15.0


def test_cli_accepts_float32_dtype_alias():
    module = load_service_module()

    args = module.build_arg_parser().parse_args(["--dtype", "float32"])

    assert args.dtype == "fp32"


def test_websocket_keepalive_kwargs_are_configurable():
    module = load_service_module()

    args = module.build_arg_parser().parse_args(
        [
            "--ws-ping-interval", "35",
            "--ws-ping-timeout", "70",
            "--ws-close-timeout", "12",
            "--ws-max-size", "12345",
        ]
    )

    assert module.build_websocket_serve_kwargs(args) == {
        "max_size": 12345,
        "ping_interval": 35.0,
        "ping_timeout": 70.0,
        "close_timeout": 12.0,
    }


def test_websocket_keepalive_kwargs_can_disable_ping():
    module = load_service_module()

    args = module.build_arg_parser().parse_args(
        [
            "--ws-ping-interval", "0",
            "--ws-ping-timeout", "0",
        ]
    )

    kwargs = module.build_websocket_serve_kwargs(args)
    assert kwargs["ping_interval"] is None
    assert kwargs["ping_timeout"] is None


def test_create_speaker_tracker_skips_spk_when_disabled():
    module = load_service_module()
    args = types.SimpleNamespace(enable_spk=False, device="cuda:0")

    assert module.create_speaker_tracker(object(), args) is None


def test_partial_decode_audio_uses_recent_window():
    module = load_service_module()

    class DummyVad:
        current_speech_start = 0

        def reset(self):
            pass

    sample_rate = 16000
    session = module.RealtimeASRSession(
        vllm_engine=object(),
        asr_kwargs={},
        vad=DummyVad(),
        spk_tracker=None,
        sample_rate=sample_rate,
        chunk_ms=960,
        partial_window_sec=2.0,
    )
    session.audio_buffer = np.arange(sample_rate * 5, dtype=np.float32)

    audio, start_ms = session.get_partial_decode_audio()

    assert start_ms == 3000
    np.testing.assert_array_equal(audio, session.audio_buffer[-sample_rate * 2:])


def test_partial_decode_audio_keeps_short_current_segment():
    module = load_service_module()

    class DummyVad:
        current_speech_start = 1500

        def reset(self):
            pass

    sample_rate = 16000
    session = module.RealtimeASRSession(
        vllm_engine=object(),
        asr_kwargs={},
        vad=DummyVad(),
        spk_tracker=None,
        sample_rate=sample_rate,
        chunk_ms=960,
        partial_window_sec=10.0,
    )
    session.audio_buffer = np.arange(sample_rate * 5, dtype=np.float32)

    audio, start_ms = session.get_partial_decode_audio()
    assert start_ms == 1500
    np.testing.assert_array_equal(audio, session.audio_buffer[int(sample_rate * 1.5):])


def test_empty_partial_response_preserves_zero_speech_start():
    module = load_service_module()

    class DummyVad:
        current_speech_start = 0

        def reset(self):
            pass

    sample_rate = 16000
    session = module.RealtimeASRSession(
        vllm_engine=object(),
        asr_kwargs={},
        vad=DummyVad(),
        spk_tracker=None,
        sample_rate=sample_rate,
        chunk_ms=960,
    )
    session.audio_buffer = np.zeros(sample_rate, dtype=np.float32)
    session.last_partial_text = ""

    response = session._build_response(is_final=False)

    assert response["partial_start_ms"] == 0
