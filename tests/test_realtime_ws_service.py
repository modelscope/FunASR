import asyncio
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import torch


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
    session.total_samples = len(session.audio_buffer)

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
    session.total_samples = len(session.audio_buffer)

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
    session.total_samples = len(session.audio_buffer)
    session.last_partial_text = ""

    response = session._build_response(is_final=False)

    assert response["partial_start_ms"] == 0


class SilentVad:
    current_speech_start = None

    def feed(self, audio, is_final=False):
        return []

    def reset(self):
        pass


class SegmentVad(SilentVad):
    def __init__(self, sample_rate, segment_end_sample):
        self.sample_rate = sample_rate
        self.segment_end_sample = segment_end_sample
        self.total_samples = 0

    def feed(self, audio, is_final=False):
        self.total_samples += len(audio)
        if self.total_samples == self.segment_end_sample:
            end_ms = int(self.total_samples * 1000 / self.sample_rate)
            return [[end_ms - 1000, end_ms]]
        return []


class DummyTokenizer:
    def encode(self, text):
        return []

    def decode(self, token_ids, skip_special_tokens=True):
        return ""


class DummyEngine:
    def __init__(self):
        self._engine = types.SimpleNamespace(tokenizer=DummyTokenizer())
        self.input_lengths = []

    def generate(self, inputs, **kwargs):
        self.input_lengths.extend(len(audio) for audio in inputs)
        return [{"text": "hello"}]


def test_two_hour_session_keeps_audio_bounded_and_duration_absolute():
    module = load_service_module()
    sample_rate = 10
    lookback_seconds = 5
    session = module.RealtimeASRSession(
        vllm_engine=DummyEngine(),
        asr_kwargs={},
        vad=SilentVad(),
        sample_rate=sample_rate,
        chunk_ms=1000,
        audio_lookback_sec=lookback_seconds,
    )
    one_second = np.zeros(sample_rate, dtype=np.int16).tobytes()

    for _ in range(2 * 60 * 60):
        session.add_audio(one_second)

    assert session.total_samples == 2 * 60 * 60 * sample_rate
    assert len(session.audio_buffer) <= lookback_seconds * sample_rate
    assert session.audio_buffer_start_sample == session.total_samples - len(session.audio_buffer)
    assert session._build_response(is_final=False)["duration_ms"] == 2 * 60 * 60 * 1000


def test_completed_segment_uses_absolute_offsets_after_audio_compaction():
    module = load_service_module()
    sample_rate = 16000
    vad = SegmentVad(sample_rate=sample_rate, segment_end_sample=11 * sample_rate)
    engine = DummyEngine()
    session = module.RealtimeASRSession(
        vllm_engine=engine,
        asr_kwargs={},
        vad=vad,
        sample_rate=sample_rate,
        chunk_ms=1000,
        audio_lookback_sec=2,
    )
    one_second = np.zeros(sample_rate, dtype=np.int16).tobytes()

    for _ in range(11):
        session.add_audio(one_second)

    assert engine.input_lengths == [sample_rate]
    assert session.locked_sentences == [{"text": "hello", "start": 10000, "end": 11000}]
    assert session._build_response(is_final=False)["duration_ms"] == 11000
    assert session.audio_buffer_start_sample > 0


def test_final_decode_releases_audio_without_resetting_absolute_duration():
    module = load_service_module()
    sample_rate = 10
    for sample_count in (5, 100):
        session = module.RealtimeASRSession(
            vllm_engine=DummyEngine(),
            asr_kwargs={},
            vad=SilentVad(),
            sample_rate=sample_rate,
            chunk_ms=1000,
            audio_lookback_sec=5,
        )
        session.add_audio(np.zeros(sample_count, dtype=np.int16).tobytes())

        response = session.decode(is_final=True)

        assert response["duration_ms"] == sample_count * 100
        assert len(session.audio_buffer) == 0
        assert session.audio_buffer_start_sample == session.total_samples


def test_speaker_history_and_identity_state_have_hard_limits(monkeypatch):
    utils_stub = types.ModuleType("funasr.models.campplus.utils")
    utils_stub.sv_chunk = lambda segments: segments

    def postprocess(segments, vad_segments, labels, embeddings, return_spk_center=False):
        output = [[segment[0], segment[1], int(label)] for segment, label in zip(segments, labels)]
        centers = torch.stack(
            [embeddings[labels == label].mean(0) for label in sorted(set(labels.tolist()))]
        )
        return (output, centers) if return_spk_center else output

    def distribute_spk(sentences, speaker_segments):
        for sentence in sentences:
            sentence["spk"] = int(speaker_segments[-1][2])
        return sentences

    utils_stub.postprocess = postprocess
    utils_stub.distribute_spk = distribute_spk
    cluster_stub = types.ModuleType("funasr.models.campplus.cluster_backend")

    class FakeClusterBackend:
        def __init__(self, merge_thr):
            pass

        def to(self, device):
            return self

        def __call__(self, embeddings, oracle_num=None):
            return np.zeros(len(embeddings), dtype=np.int64)

    cluster_stub.ClusterBackend = FakeClusterBackend
    monkeypatch.setitem(sys.modules, "funasr.models.campplus.utils", utils_stub)
    monkeypatch.setitem(sys.modules, "funasr.models.campplus.cluster_backend", cluster_stub)

    module = load_service_module()

    class FakeSpeakerModel:
        def generate(self, input, **kwargs):
            return [{"spk_embedding": torch.tensor([[1.0, 0.0]])} for _ in input]

    tracker = module.HybridSpeakerTracker(
        spk_model=FakeSpeakerModel(),
        device="cpu",
        max_history_chunks=3,
        max_speakers=2,
    )
    tracker.sv_chunk = lambda segments: [
        [segments[0][0], segments[0][1], segments[0][2]]
    ]
    tracker.cluster_backend = lambda embeddings, oracle_num=None: np.zeros(
        len(embeddings), dtype=np.int64
    )

    for index in range(6):
        sentence = {"text": f"segment {index}", "start": index * 1000, "end": (index + 1) * 1000}
        tracker.assign_streaming(
            np.ones(16000, dtype=np.float32),
            index,
            index + 1,
            sentence,
        )
        assert sentence["spk"] == 0

    assert len(tracker.all_chunks) == 3
    assert len(tracker.all_embeddings) == 3
    assert all(len(chunk) == 2 for chunk in tracker.all_chunks)

    first = tracker._map_cluster_centers(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), update=True
    )
    swapped = tracker._map_cluster_centers(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]), update=True
    )
    capped = tracker._map_cluster_centers(
        torch.tensor([[-1.0, 0.0]]), update=True
    )

    assert first == [0, 1]
    assert swapped == [1, 0]
    assert capped[0] in {0, 1}
    assert len(tracker.speaker_centers) == 2
    assert tracker.all_chunks.maxlen == 3
    assert tracker.all_embeddings.maxlen == 3

    finalized = tracker.finalize(
        [{"text": "abcdef", "start": 0, "end": 6000, "spk": 1}]
    )
    assert finalized == [
        {"text": "abc", "start": 0, "end": 3000, "spk": 1},
        {"text": "def", "start": 3000, "end": 6000, "spk": 0},
    ]


def test_handler_keeps_event_loop_responsive_during_session_work(monkeypatch):
    module = load_service_module()

    class BlockingSession:
        active_workers = 0
        max_active_workers = 0
        worker_lock = threading.Lock()

        def __init__(self, *args, **kwargs):
            self.is_active = False
            self.audio_buffer = np.zeros(1, dtype=np.float32)

        def reset(self):
            pass

        def add_audio(self, message):
            with self.worker_lock:
                type(self).active_workers += 1
                type(self).max_active_workers = max(
                    type(self).max_active_workers,
                    type(self).active_workers,
                )
            try:
                time.sleep(0.2)
            finally:
                with self.worker_lock:
                    type(self).active_workers -= 1

        def should_decode(self):
            return False

    class FakeWebSocket:
        remote_address = ("127.0.0.1", 12345)

        def __init__(self):
            self.messages = iter(["START", b"audio"])
            self.sent = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.messages)
            except StopIteration as error:
                raise StopAsyncIteration from error

        async def send(self, message):
            self.sent.append(message)

    monkeypatch.setattr(module, "load_models", lambda args: (object(), {}, object(), None))
    monkeypatch.setattr(module, "DynamicStreamingVAD", lambda model: object())
    monkeypatch.setattr(module, "create_speaker_tracker", lambda model, args: None)
    monkeypatch.setattr(module, "RealtimeASRSession", BlockingSession)

    async def exercise_handler():
        stop = False
        gaps = []

        async def ticker():
            previous = asyncio.get_running_loop().time()
            while not stop:
                await asyncio.sleep(0.005)
                current = asyncio.get_running_loop().time()
                gaps.append(current - previous)
                previous = current

        ticker_task = asyncio.create_task(ticker())
        await asyncio.sleep(0.01)
        args = types.SimpleNamespace(device="cpu", decode_interval=0.48, partial_window_sec=15.0)
        await asyncio.gather(
            module.handle_client(FakeWebSocket(), args),
            module.handle_client(FakeWebSocket(), args),
        )
        stop = True
        await ticker_task
        return gaps

    gaps = asyncio.run(exercise_handler())

    assert gaps
    assert max(gaps) < 0.08
    assert BlockingSession.max_active_workers == 1
