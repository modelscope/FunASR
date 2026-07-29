import ast
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "runtime/python/websocket/funasr_wss_server.py"
CLIENT_PATH = ROOT / "runtime/python/websocket/funasr_wss_client.py"
README_PATH = ROOT / "runtime/python/websocket/README.md"


def _load_async_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


class FakeConnectionClosed(Exception):
    pass


class FakeInvalidState(Exception):
    pass


class FakeWebSocket:
    def __init__(self, incoming=()):
        self.incoming = iter(incoming)
        self.sent = []
        self.closed = False
        self.path = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.incoming)
        except StopIteration:
            raise StopAsyncIteration

    async def send(self, message):
        self.sent.append(message)

    async def close(self):
        self.closed = True


def _server_namespace(
    asr_calls,
    online_calls,
    vad_results=None,
    asr_error=None,
    online_error=None,
    asr_error_once=None,
    online_error_once=None,
    online_cache_states=None,
    reset_calls=None,
    vad_calls=None,
):
    vad_results = iter(vad_results or [])
    online_cache_states = online_cache_states if online_cache_states is not None else []
    reset_calls = reset_calls if reset_calls is not None else []
    vad_calls = vad_calls if vad_calls is not None else []

    async def async_vad(websocket, pcm):
        vad_calls.append(pcm)
        return next(vad_results, (0, -1))

    async def async_asr(websocket, audio):
        asr_calls.append(audio)
        if asr_error is not None:
            raise asr_error
        if asr_error_once is not None and len(asr_calls) == 1:
            raise asr_error_once
        await websocket.send(
            json.dumps(
                {
                    "mode": (
                        "2pass-offline" if websocket.mode == "2pass" else websocket.mode
                    ),
                    "text": "offline final",
                    "wav_name": websocket.wav_name,
                    "is_final": True,
                }
            )
        )

    async def async_asr_online(websocket, audio):
        online_calls.append(audio)
        online_cache_states.append(dict(websocket.status_dict_asr_online["cache"]))
        if online_error is not None:
            raise online_error
        if online_error_once is not None and len(online_calls) == 1:
            raise online_error_once
        if not websocket.status_dict_asr_online.get("is_final", False):
            websocket.status_dict_asr_online["cache"]["partial"] = True
        await websocket.send(
            json.dumps(
                {
                    "mode": (
                        "2pass-online" if websocket.mode == "2pass" else websocket.mode
                    ),
                    "text": "online final",
                    "wav_name": websocket.wav_name,
                    "is_final": True,
                }
            )
        )

    async def ws_reset(websocket):
        reset_calls.append(websocket)
        await websocket.close()

    return {
        "asyncio": asyncio,
        "json": json,
        "websockets": SimpleNamespace(
            ConnectionClosed=FakeConnectionClosed,
            InvalidState=FakeInvalidState,
        ),
        "websocket_users": set(),
        "args": SimpleNamespace(
            save_offline_segments=False, save_offline_segments_dir="unused"
        ),
        "_safe_int": lambda value, default: (
            int(value) if value is not None else default
        ),
        "_pcm_duration_ms": lambda pcm, fs, ch, sampwidth: 10,
        "async_vad": async_vad,
        "async_asr": async_asr,
        "async_asr_online": async_asr_online,
        "run_blocking": None,
        "save_offline_wav_segment_sync": None,
        "SEM_WAV": None,
        "ws_reset": ws_reset,
    }


def _run_server(
    mode,
    frame_count=1,
    chunk_interval=10,
    chunk_size_value=(5, 10, 5),
    send_end_ack=True,
    vad_results=None,
    asr_error=None,
    online_error=None,
    asr_error_once=None,
    online_error_once=None,
    online_cache_states=None,
    vad_calls=None,
):
    audio = b"\x01\x00" * 160
    config = {
        "mode": mode,
        "chunk_interval": chunk_interval,
        "audio_fs": 16000,
        "wav_name": "sample",
        "is_speaking": True,
    }
    if chunk_size_value is not None:
        config["chunk_size"] = list(chunk_size_value)
    end_message = {"is_speaking": False}
    if send_end_ack:
        end_message["is_end"] = True
    incoming = [
        json.dumps(config),
        *([audio] * frame_count),
        json.dumps(end_message),
    ]
    websocket = FakeWebSocket(incoming)
    asr_calls = []
    online_calls = []
    namespace = _server_namespace(
        asr_calls,
        online_calls,
        vad_results=vad_results,
        asr_error=asr_error,
        online_error=online_error,
        asr_error_once=asr_error_once,
        online_error_once=online_error_once,
        online_cache_states=online_cache_states,
        vad_calls=vad_calls,
    )
    ws_serve = _load_async_function(SERVER_PATH, "ws_serve", namespace)
    asyncio.run(ws_serve(websocket))
    sent = [json.loads(message) for message in websocket.sent]
    return audio, asr_calls, online_calls, sent


def test_offline_end_control_flushes_buffer_before_acknowledgement():
    audio, asr_calls, online_calls, sent = _run_server("offline")

    assert asr_calls == [audio]
    assert online_calls == []
    assert sent[-1] == {
        "mode": "offline",
        "wav_name": "sample",
        "is_final": True,
        "is_end": True,
    }


def test_online_end_control_flushes_residual_frames_before_acknowledgement():
    audio, asr_calls, online_calls, sent = _run_server("online")

    assert asr_calls == []
    assert online_calls == [audio]
    assert sent[-1] == {
        "mode": "online",
        "wav_name": "sample",
        "is_final": True,
        "is_end": True,
    }


def test_2pass_end_control_flushes_online_and_offline_before_acknowledgement():
    audio, asr_calls, online_calls, sent = _run_server("2pass")

    assert online_calls == [audio]
    assert asr_calls == [audio]
    assert sent[-1] == {
        "mode": "2pass",
        "wav_name": "sample",
        "is_final": True,
        "is_end": True,
    }


def test_online_finalizes_cache_after_exact_chunk_boundary():
    audio, asr_calls, online_calls, sent = _run_server(
        "online", frame_count=2, chunk_interval=2
    )

    assert asr_calls == []
    assert online_calls == [audio * 2, b""]
    assert sent[-1]["is_end"] is True


def test_online_vad_endpoint_flushes_residual_audio_before_cache_reset():
    audio, asr_calls, online_calls, sent = _run_server("online", vad_results=[(0, 10)])

    assert asr_calls == []
    assert online_calls == [audio]
    assert sent[-1]["is_end"] is True


def test_online_vad_endpoint_flushes_exact_chunk_cache_before_reset():
    cache_states = []
    audio, asr_calls, online_calls, sent = _run_server(
        "online",
        chunk_interval=1,
        vad_results=[(0, 10)],
        online_cache_states=cache_states,
    )

    assert asr_calls == []
    assert online_calls == [audio, b""]
    assert cache_states == [{}, {"partial": True}]
    assert sent[-1]["is_end"] is True


def test_offline_end_uses_buffered_audio_when_vad_has_not_started():
    audio, asr_calls, online_calls, sent = _run_server(
        "offline", vad_results=[(-1, -1)]
    )

    assert online_calls == []
    assert asr_calls == [audio]
    assert sent[-1]["is_end"] is True


def test_offline_end_does_not_emit_empty_final_after_vad_completion():
    audio, asr_calls, online_calls, sent = _run_server("offline", vad_results=[(0, 10)])

    assert online_calls == []
    assert asr_calls == [audio]
    assert sent[-1]["is_end"] is True


def test_offline_end_flushes_tail_after_an_earlier_vad_completion():
    audio, asr_calls, online_calls, sent = _run_server(
        "offline", frame_count=2, vad_results=[(0, 10), (-1, -1)]
    )

    assert online_calls == []
    assert asr_calls == [audio, audio]
    assert sent[-1]["is_end"] is True


def test_offline_end_falls_back_when_vad_reports_only_an_endpoint():
    audio, asr_calls, online_calls, sent = _run_server(
        "offline", vad_results=[(-1, 10)]
    )

    assert online_calls == []
    assert asr_calls == [audio]
    assert sent[-1]["is_end"] is True


def test_offline_endpoint_only_audio_is_not_dropped_by_a_later_segment():
    audio, asr_calls, online_calls, sent = _run_server(
        "offline",
        frame_count=3,
        vad_results=[(-1, 10), (10, -1), (-1, 30)],
    )

    assert online_calls == []
    assert asr_calls == [audio, audio * 2]
    assert sent[-1]["is_end"] is True


def test_end_ack_reports_audio_discarded_without_chunk_size():
    _, asr_calls, online_calls, sent = _run_server("offline", chunk_size_value=None)

    assert asr_calls == []
    assert online_calls == []
    assert sent[-1]["is_end"] is True
    assert sent[-1]["is_final"] is False
    assert "chunk_size" in sent[-1]["error"]


def test_end_ack_reports_audio_discarded_for_invalid_vad_chunk_size():
    _, asr_calls, online_calls, sent = _run_server("offline", chunk_size_value=(5,))

    assert asr_calls == []
    assert online_calls == []
    assert sent[-1]["is_end"] is True
    assert sent[-1]["is_final"] is False
    assert "VAD chunk_size" in sent[-1]["error"]


def test_end_ack_rejects_unsupported_mode():
    vad_calls = []
    _, asr_calls, online_calls, sent = _run_server("bogus", vad_calls=vad_calls)

    assert vad_calls == []
    assert asr_calls == []
    assert online_calls == []
    assert sent[-1]["is_end"] is True
    assert sent[-1]["is_final"] is False
    assert "unsupported mode" in sent[-1]["error"]


def test_invalid_state_resets_and_unregisters_connection():
    audio = b"\x01\x00" * 160
    incoming = [
        json.dumps(
            {
                "mode": "offline",
                "chunk_interval": 10,
                "audio_fs": 16000,
                "wav_name": "sample",
                "is_speaking": True,
                "chunk_size": [5, 10, 5],
            }
        ),
        audio,
        json.dumps({"is_speaking": False, "is_end": True}),
    ]

    class InvalidStateWebSocket(FakeWebSocket):
        async def send(self, message):
            raise FakeInvalidState

    websocket = InvalidStateWebSocket(incoming)
    reset_calls = []
    namespace = _server_namespace([], [], reset_calls=reset_calls)
    ws_serve = _load_async_function(SERVER_PATH, "ws_serve", namespace)

    asyncio.run(ws_serve(websocket))

    assert reset_calls == [websocket]
    assert websocket.closed is True
    assert websocket not in namespace["websocket_users"]


def test_old_end_control_without_is_end_remains_supported():
    audio, asr_calls, online_calls, sent = _run_server("offline", send_end_ack=False)

    assert asr_calls == [audio]
    assert online_calls == []
    assert not any(message.get("is_end") for message in sent)


def test_end_ack_reports_inference_failure():
    _, asr_calls, _, sent = _run_server(
        "offline", asr_error=RuntimeError("offline inference failed")
    )

    assert len(asr_calls) == 1
    assert sent[-1]["is_end"] is True
    assert sent[-1]["is_final"] is False
    assert "offline inference failed" in sent[-1]["error"]


def test_end_ack_preserves_earlier_online_inference_failure():
    _, _, online_calls, sent = _run_server(
        "online",
        chunk_interval=1,
        online_error_once=RuntimeError("streaming chunk failed"),
    )

    assert len(online_calls) == 2
    assert sent[-1]["is_end"] is True
    assert sent[-1]["is_final"] is False
    assert "streaming chunk failed" in sent[-1]["error"]


def test_end_ack_preserves_earlier_offline_inference_failure():
    _, asr_calls, _, sent = _run_server(
        "offline",
        vad_results=[(0, 10)],
        asr_error_once=RuntimeError("offline segment failed"),
    )

    assert len(asr_calls) == 2
    assert sent[-1]["is_end"] is True
    assert sent[-1]["is_final"] is False
    assert "offline segment failed" in sent[-1]["error"]


def test_online_final_with_empty_audio_still_flushes_model_cache():
    calls = []

    async def run_blocking(function, model, audio, status, sem):
        calls.append((function, model, audio, status, sem))
        return [{"text": ""}]

    namespace = {
        "json": json,
        "run_blocking": run_blocking,
        "_generate_sync": object(),
        "model_asr_streaming": object(),
        "SEM_ASR_ONLINE": object(),
    }
    async_asr_online = _load_async_function(SERVER_PATH, "async_asr_online", namespace)
    websocket = FakeWebSocket()
    websocket.mode = "online"
    websocket.wav_name = "sample"
    websocket.is_speaking = False
    websocket.status_dict_asr_online = {"cache": {"partial": True}, "is_final": True}

    asyncio.run(async_asr_online(websocket, b""))

    assert len(calls) == 1
    assert calls[0][2] == b""


def test_offline_result_send_failure_propagates_to_the_caller():
    generate_sync = object()
    speaker_sync = object()

    async def run_blocking(function, *args, sem):
        if function is generate_sync:
            return [{"text": "final text"}]
        if function is speaker_sync:
            return "unknown", 0.0
        raise AssertionError("unexpected blocking function")

    namespace = {
        "json": json,
        "run_blocking": run_blocking,
        "_generate_sync": generate_sync,
        "_sv_and_match_sync": speaker_sync,
        "model_asr": object(),
        "model_punc": None,
        "SEM_ASR_OFFLINE": object(),
        "SEM_SV": object(),
        "args": SimpleNamespace(speaker_db_reload_sec=60),
        "to_python": lambda value: value,
    }
    async_asr = _load_async_function(SERVER_PATH, "async_asr", namespace)

    class FailingResultWebSocket(FakeWebSocket):
        async def send(self, message):
            raise RuntimeError("result send failed")

    websocket = FailingResultWebSocket()
    websocket.mode = "offline"
    websocket.wav_name = "sample"
    websocket.status_dict_asr = {}
    websocket.status_dict_punc = {"cache": {}}

    with pytest.raises(RuntimeError, match="result send failed"):
        asyncio.run(async_asr(websocket, b"\x01\x00"))


def _client_namespace(args, websocket):
    return {
        "asyncio": asyncio,
        "json": json,
        "os": __import__("os"),
        "time": __import__("time"),
        "args": args,
        "websocket": websocket,
        "voices": None,
        "offline_msg_done": False,
        "latency_first_audio_time": {},
        "latency_last_audio_time": {},
        "latency_first_text_printed": {},
    }


def test_file_sender_waits_for_protocol_ack_instead_of_fixed_sleep(tmp_path):
    pcm_path = tmp_path / "sample.pcm"
    pcm_path.write_bytes(b"\x01\x00" * 160)
    websocket = FakeWebSocket()
    args = SimpleNamespace(
        audio_in=str(pcm_path),
        hotword="",
        audio_fs=16000,
        use_itn=1,
        chunk_size=[5, 10, 5],
        chunk_interval=10,
        encoder_chunk_look_back=4,
        decoder_chunk_look_back=0,
        mode="offline",
        send_without_sleep=True,
        result_timeout=1.0,
    )
    namespace = _client_namespace(args, websocket)
    record_from_scp = _load_async_function(CLIENT_PATH, "record_from_scp", namespace)

    async def exercise():
        completed = asyncio.Queue()
        task = asyncio.create_task(record_from_scp(0, 1, completed))
        try:
            for _ in range(100):
                if any(
                    isinstance(message, str) and json.loads(message).get("is_end")
                    for message in websocket.sent
                ):
                    break
                await asyncio.sleep(0)

            assert not task.done()
            completed.put_nowait({"is_end": True, "is_final": True})
            await asyncio.wait_for(task, timeout=0.2)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())

    end_message = json.loads(websocket.sent[-1])
    assert end_message == {"is_speaking": False, "is_end": True}
    assert websocket.closed is True


def test_file_sender_surfaces_server_error_ack(tmp_path):
    pcm_path = tmp_path / "sample.pcm"
    pcm_path.write_bytes(b"\x01\x00" * 160)
    websocket = FakeWebSocket()
    args = SimpleNamespace(
        audio_in=str(pcm_path),
        hotword="",
        audio_fs=16000,
        use_itn=1,
        chunk_size=[5, 10, 5],
        chunk_interval=10,
        encoder_chunk_look_back=4,
        decoder_chunk_look_back=0,
        mode="offline",
        send_without_sleep=True,
        result_timeout=1.0,
    )
    namespace = _client_namespace(args, websocket)
    record_from_scp = _load_async_function(CLIENT_PATH, "record_from_scp", namespace)

    async def exercise():
        completed = asyncio.Queue()
        task = asyncio.create_task(record_from_scp(0, 1, completed))
        try:
            for _ in range(100):
                if any(
                    isinstance(message, str) and json.loads(message).get("is_end")
                    for message in websocket.sent
                ):
                    break
                await asyncio.sleep(0)

            completed.put_nowait(
                {"is_end": True, "is_final": False, "error": "offline inference failed"}
            )
            with pytest.raises(RuntimeError, match="offline inference failed"):
                await asyncio.wait_for(task, timeout=0.2)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_file_sender_rejects_nonfinal_ack_without_error(tmp_path):
    pcm_path = tmp_path / "sample.pcm"
    pcm_path.write_bytes(b"\x01\x00" * 160)
    websocket = FakeWebSocket()
    args = SimpleNamespace(
        audio_in=str(pcm_path),
        hotword="",
        audio_fs=16000,
        use_itn=1,
        chunk_size=[5, 10, 5],
        chunk_interval=10,
        encoder_chunk_look_back=4,
        decoder_chunk_look_back=0,
        mode="offline",
        send_without_sleep=True,
        result_timeout=1.0,
    )
    namespace = _client_namespace(args, websocket)
    record_from_scp = _load_async_function(CLIENT_PATH, "record_from_scp", namespace)

    async def exercise():
        completed = asyncio.Queue()
        task = asyncio.create_task(record_from_scp(0, 1, completed))
        try:
            for _ in range(100):
                if any(
                    isinstance(message, str) and json.loads(message).get("is_end")
                    for message in websocket.sent
                ):
                    break
                await asyncio.sleep(0)

            completed.put_nowait({"is_end": True, "is_final": False})
            with pytest.raises(RuntimeError, match="did not finalize"):
                await asyncio.wait_for(task, timeout=0.2)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_file_sender_closes_connection_after_result_timeout(tmp_path):
    pcm_path = tmp_path / "sample.pcm"
    pcm_path.write_bytes(b"\x01\x00" * 160)
    websocket = FakeWebSocket()
    args = SimpleNamespace(
        audio_in=str(pcm_path),
        hotword="",
        audio_fs=16000,
        use_itn=1,
        chunk_size=[5, 10, 5],
        chunk_interval=10,
        encoder_chunk_look_back=4,
        decoder_chunk_look_back=0,
        mode="offline",
        send_without_sleep=True,
        result_timeout=0.01,
    )
    namespace = _client_namespace(args, websocket)
    record_from_scp = _load_async_function(CLIENT_PATH, "record_from_scp", namespace)

    async def exercise():
        completed = asyncio.Queue()
        with pytest.raises(TimeoutError, match="did not acknowledge"):
            await record_from_scp(0, 1, completed)

    asyncio.run(exercise())

    assert websocket.closed is True


def test_receiver_releases_file_sender_only_on_end_acknowledgement(monkeypatch):
    class ReceivingWebSocket(FakeWebSocket):
        async def recv(self):
            try:
                return next(self.incoming)
            except StopIteration:
                raise FakeConnectionClosed

    websocket = ReceivingWebSocket(
        [
            json.dumps(
                {"mode": "offline", "text": "done", "is_final": True, "is_end": True}
            )
        ]
    )
    args = SimpleNamespace(thread_num=1, output_dir=None, words_max_print=10000)
    namespace = _client_namespace(args, websocket)
    namespace.update(
        {
            "websockets": SimpleNamespace(
                exceptions=SimpleNamespace(ConnectionClosedOK=FakeConnectionClosed)
            ),
            "MeetingWriter": object,
            "_iso": lambda ts: str(ts),
        }
    )
    message = _load_async_function(CLIENT_PATH, "message", namespace)
    monkeypatch.setitem(sys.modules, "websockets", namespace["websockets"])

    async def exercise():
        completed = asyncio.Queue()
        await message("0", None, completed)
        acknowledgement = completed.get_nowait()
        assert acknowledgement["is_end"] is True

    asyncio.run(exercise())


def test_receiver_failure_releases_waiting_file_sender(monkeypatch):
    class ReceivingWebSocket(FakeWebSocket):
        async def recv(self):
            try:
                return next(self.incoming)
            except StopIteration:
                raise FakeConnectionClosed

    websocket = ReceivingWebSocket(["not-json"])
    args = SimpleNamespace(thread_num=1, output_dir=None, words_max_print=10000)
    namespace = _client_namespace(args, websocket)
    namespace.update(
        {
            "websockets": SimpleNamespace(
                exceptions=SimpleNamespace(ConnectionClosedOK=FakeConnectionClosed)
            ),
            "MeetingWriter": object,
            "_iso": lambda ts: str(ts),
        }
    )
    message = _load_async_function(CLIENT_PATH, "message", namespace)
    monkeypatch.setitem(sys.modules, "websockets", namespace["websockets"])

    async def exercise():
        completed = asyncio.Queue()
        await message("0", None, completed)
        failure = completed.get_nowait()
        assert "receiver failed" in failure["error"]

    asyncio.run(exercise())


def test_readme_documents_file_completion_acknowledgement():
    readme = README_PATH.read_text(encoding="utf-8")

    assert "--result_timeout" in readme
    assert '"is_speaking": false, "is_end": true' in readme
    assert '"is_end": true, "is_final": true' in readme
    assert '"is_final": false, "error"' in readme
