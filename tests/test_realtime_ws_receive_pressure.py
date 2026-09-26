"""Bounded receive ownership, with synthetic acoustics and real WebSockets."""

import asyncio
import gc
import json
import threading
import types
import weakref

import pytest

from test_realtime_ws_service import load_service_module


async def until(predicate, timeout=3):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout)


class Socket:
    remote_address = ("127.0.0.1", 1)

    def __init__(self, messages=(), *, error=None, close_code=None, hold=False):
        self.messages = iter(messages)
        self.error = error
        self.close_code = None
        self.end_close_code = close_code
        self.hold = hold
        self.exhausted = False
        self.cancelled = False
        self.closed = []
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.messages)
        except StopIteration:
            self.exhausted = True
            self.close_code = self.end_close_code
        if self.hold:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.error:
            raise self.error
        raise StopAsyncIteration

    async def close(self, code=1000, reason=""):
        self.closed.append((code, reason))
        self.close_code = code

    async def send(self, message):
        self.sent.append(json.loads(message))


def buffer_class(module):
    assert hasattr(module, "RealtimeReceiveBuffer"), "missing bounded receive owner"
    assert hasattr(module, "ReceiveBufferOverflow"), "missing explicit overload failure"
    return module.RealtimeReceiveBuffer


def test_fifo_exact_limits_utf8_and_command_head():
    module = load_service_module()
    factory = buffer_class(module)

    async def exercise():
        messages = [b"ab", "中", b"", b"z", "STOP"]
        socket = Socket(messages)
        buffer = factory(socket, max_messages=5, max_bytes=10)
        buffer.start()
        try:
            await until(lambda: socket.exhausted)
            for expected, pending in zip(messages, [True, False, False, True, False]):
                assert buffer.pending_audio is pending
                assert await asyncio.wait_for(buffer.__anext__(), 1) == expected
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(buffer.__anext__(), 1)
            assert socket.closed == []
        finally:
            await buffer.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "messages,limits",
    [([b"a", "STOP"], {"max_messages": 1, "max_bytes": 100}),
     (["中"], {"max_messages": 8, "max_bytes": 2}),
     ([b"abc"], {"max_messages": 8, "max_bytes": 2})],
)
def test_overflow_discards_queue_and_closes_1013(messages, limits):
    module = load_service_module()
    factory = buffer_class(module)

    async def exercise():
        socket = Socket(messages)
        buffer = factory(socket, **limits)
        buffer.start()
        try:
            await until(lambda: socket.closed)
            assert socket.closed[0][0] == 1013
            assert 0 < len(socket.closed[0][1].encode("utf-8")) <= 123
            assert buffer.pending_audio is False
            with pytest.raises(module.ReceiveBufferOverflow):
                await asyncio.wait_for(buffer.__anext__(), 1)
        finally:
            await buffer.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("name", ["max_messages", "max_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_buffer_rejects_nonpositive_or_noninteger_limits(name, value):
    module = load_service_module()
    factory = buffer_class(module)
    with pytest.raises((ValueError, TypeError)):
        factory(Socket(), **{name: value})


@pytest.mark.parametrize("mode", ["peer_close", "receive_error"])
def test_disconnect_discards_accepted_work(mode):
    module = load_service_module()
    factory = buffer_class(module)

    async def exercise():
        socket = Socket([b"pcm", "COMMIT"], close_code=1000 if mode == "peer_close" else None,
                        error=RuntimeError("receive failed") if mode == "receive_error" else None)
        buffer = factory(socket)
        buffer.start()
        try:
            await until(lambda: socket.exhausted)
            assert buffer.pending_audio is False
            try:
                value = await asyncio.wait_for(buffer.__anext__(), 1)
            except (StopAsyncIteration, RuntimeError):
                pass
            else:
                pytest.fail(f"disconnected input was dispatched: {value!r}")
        finally:
            await buffer.aclose()

    asyncio.run(exercise())


def test_aclose_joins_reader_and_releases_pending_input():
    module = load_service_module()
    factory = buffer_class(module)

    async def exercise():
        socket = Socket([b"pcm"], hold=True)
        buffer = factory(socket)
        buffer.start()
        await until(lambda: socket.exhausted)
        await asyncio.wait_for(buffer.aclose(), 1)
        assert socket.cancelled
        assert buffer.pending_audio is False
        await asyncio.wait_for(buffer.aclose(), 1)

    asyncio.run(exercise())


def test_aclose_releases_queued_payload_references():
    module = load_service_module()
    factory = buffer_class(module)

    class TextPayload(str):
        pass

    async def exercise():
        payload = TextPayload("queued command payload")
        reference = weakref.ref(payload)
        socket = Socket([payload], hold=True)
        del payload
        buffer = factory(socket)
        buffer.start()
        try:
            await until(lambda: socket.exhausted)
            assert reference() is not None
        finally:
            await asyncio.wait_for(buffer.aclose(), 1)
        # Let completed-task callbacks release their transient references.
        await asyncio.sleep(0)
        gc.collect()
        assert reference() is None, "closed receive owner retained queued payload"

    asyncio.run(exercise())


def test_cli_receive_defaults_do_not_change_transport_defaults():
    module = load_service_module()
    args = module.build_arg_parser().parse_args([])
    assert getattr(args, "ws_receive_max_messages", None) == 128
    assert getattr(args, "ws_receive_max_bytes", None) == 16777216
    assert args.partial_window_sec == 8.0
    kwargs = module.build_websocket_serve_kwargs(args)
    assert "max_queue" not in kwargs
    assert kwargs["ping_interval"] == 20.0
    assert kwargs["ping_timeout"] is None


@pytest.mark.parametrize("flag", ["--ws-receive-max-messages", "--ws-receive-max-bytes"])
def test_cli_accepts_positive_limits_and_rejects_zero(flag):
    parser = load_service_module().build_arg_parser()
    args = parser.parse_args([flag, "7"])
    assert getattr(args, flag[2:].replace("-", "_")) == 7
    for value in ["0", "-1"]:
        with pytest.raises(SystemExit):
            parser.parse_args([flag, value])


def test_cancellation_does_not_return_before_worker_stops_mutating():
    module = load_service_module()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work():
        entered.set()
        assert release.wait(5), "test failed to release owned worker"
        finished.set()

    async def exercise():
        task = asyncio.create_task(module.run_session_work(None, work))
        try:
            await until(entered.is_set)
            task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done(), "cancellation returned while thread could still mutate session"
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            await until(finished.is_set)

    asyncio.run(exercise())


@pytest.mark.parametrize("worker_error", [False, True])
def test_repeated_cancellation_joins_worker_even_when_worker_fails(worker_error):
    module = load_service_module()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work():
        entered.set()
        try:
            assert release.wait(5)
            if worker_error:
                raise RuntimeError("synthetic worker failure during cancellation")
        finally:
            finished.set()

    async def exercise():
        errors = []
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        task = asyncio.create_task(module.run_session_work(None, work))
        try:
            await until(entered.is_set)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done(), "a repeated cancellation abandoned the owned thread"
        finally:
            release.set()
            try:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
                await until(finished.is_set)
                del task
                gc.collect()
                await asyncio.sleep(0)
                assert errors == [], "worker exception was not retrieved"
            finally:
                loop.set_exception_handler(previous)

    asyncio.run(exercise())


def test_dequeue_returns_message_and_byte_credits():
    module = load_service_module()
    factory = buffer_class(module)

    async def exercise():
        class Controlled(Socket):
            def __init__(self):
                super().__init__()
                self.incoming = asyncio.Queue()
                self.requests = 0

            async def __anext__(self):
                self.requests += 1
                return await self.incoming.get()

        socket = Controlled()
        buffer = factory(socket, max_messages=1, max_bytes=3)
        buffer.start()
        try:
            for number, message in enumerate(["中", b"abc", "xyz"], start=1):
                socket.incoming.put_nowait(message)
                await until(lambda: socket.requests >= number + 1 or socket.closed)
                assert not socket.closed
                assert await asyncio.wait_for(buffer.__anext__(), 1) == message
        finally:
            await buffer.aclose()

    asyncio.run(exercise())


def session_fixture(monkeypatch, module, *, block=None, entered=None, release=None):
    sessions = []

    class Session:
        def __init__(self, engine, config, vad, **kwargs):
            self.is_active = False
            self.asr_kwargs = dict(config)
            self.total_samples = 0
            self.audio_buffer_start_sample = 0
            self.audio = []
            self.completed = []
            self.config_at_audio = []
            self.previews = []
            self.finals = []
            sessions.append(self)

        def reset(self):
            self.total_samples = 0
            self.audio_buffer_start_sample = 0

        def wait_if_needed(self, operation):
            if operation == block and self.total_samples == 1:
                entered.set()
                assert release.wait(8), "owned synthetic decoder was not released"

        def add_audio(self, message):
            self.audio.append(message)
            self.total_samples += 1
            self.config_at_audio.append(dict(self.asr_kwargs))
            self.wait_if_needed("add_audio")
            # This boundary stands in for completed-segment work inside add_audio.
            self.completed.append(message)

        def should_decode(self):
            return True

        def decode(self, is_final=False):
            self.wait_if_needed("decode")
            (self.finals if is_final else self.previews).append(self.total_samples)
            return {"event": "final" if is_final else "partial",
                    "count": self.total_samples, "sentences": []}

        def commit(self):
            self.finals.append(self.total_samples)
            self.audio_buffer_start_sample = self.total_samples
            return {"event": "final", "count": self.total_samples, "sentences": []}

    monkeypatch.setattr(module, "load_models", lambda args: (object(), {}, object(), None))
    monkeypatch.setattr(module, "create_vad", lambda *args: object())
    monkeypatch.setattr(module, "create_speaker_tracker", lambda *args: None)
    monkeypatch.setattr(module, "RealtimeASRSession", Session)
    return sessions


def handler_args(**overrides):
    values = dict(device="cpu", decode_interval=0.0, partial_window_sec=8.0,
                  endpoint_mode="client", ws_receive_max_messages=128,
                  ws_receive_max_bytes=16777216)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_handler_defers_only_obsolete_previews_preserves_commands_and_finals(monkeypatch):
    module = load_service_module()
    sessions = session_fixture(monkeypatch, module)
    socket = Socket(["START", b"a", b"b", "HOTWORDS:alpha,beta", "LANGUAGE:en",
                     "COMMIT", b"c", "STOP"])
    asyncio.run(asyncio.wait_for(module.handle_client(socket, handler_args()), 3))
    session = sessions[0]
    assert session.audio == [b"a", b"b", b"c"]
    assert session.completed == session.audio
    assert session.config_at_audio == [{}, {}, {"hotwords": ["alpha", "beta"], "language": "en"}]
    assert session.previews == [2, 3], "skip stale audio preview, not the preview before a command"
    assert session.finals == [2, 3]
    assert [item["event"] for item in socket.sent] == [
        "started", "partial", "hotwords_set", "language_set", "final", "partial", "final", "stopped"]


def test_no_backlog_keeps_each_due_preview(monkeypatch):
    module = load_service_module()
    sessions = session_fixture(monkeypatch, module)

    class Paced(Socket):
        async def __anext__(self):
            message = await super().__anext__()
            required = {b"b": 1, "STOP": 2}.get(message, 0)
            if required:
                await until(lambda: sum(x["event"] == "partial" for x in self.sent) >= required)
            return message

    socket = Paced(["START", b"a", b"b", "STOP"])
    asyncio.run(asyncio.wait_for(module.handle_client(socket, handler_args()), 4))
    assert sessions[0].previews == [1, 2]
    assert sessions[0].finals == [2]
    assert socket.sent[-1] == {"event": "stopped"}


def test_handler_failure_closes_owned_reader(monkeypatch):
    module = load_service_module()
    session_fixture(monkeypatch, module)

    class SendFailure(Socket):
        async def send(self, message):
            await until(lambda: self.exhausted)
            raise RuntimeError("synthetic send failure")

    socket = SendFailure(["START"], hold=True)
    asyncio.run(asyncio.wait_for(module.handle_client(socket, handler_args()), 4))
    assert socket.cancelled, "consumer failure leaked its receive task"


def test_handler_cancellation_waits_worker_and_joins_receiver(monkeypatch):
    module = load_service_module()
    entered, release = threading.Event(), threading.Event()
    sessions = session_fixture(monkeypatch, module, block="add_audio", entered=entered, release=release)

    async def exercise():
        socket = Socket(["START", b"a"], hold=True)
        task = asyncio.create_task(module.handle_client(socket, handler_args()))
        try:
            await until(entered.is_set)
            task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done(), "handler returned with an active session-mutating worker"
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        assert sessions[0].completed == [b"a"]
        assert socket.cancelled

    asyncio.run(exercise())


def test_handler_overflow_does_not_dispatch_queued_stop_or_success(monkeypatch):
    module = load_service_module()
    entered, release = threading.Event(), threading.Event()
    sessions = session_fixture(monkeypatch, module, block="add_audio", entered=entered, release=release)

    class Burst(Socket):
        async def __anext__(self):
            message = await super().__anext__()
            if message == b"b":
                await until(entered.is_set)
            return message

    async def exercise():
        socket = Burst(["START", b"a", b"b", b"c", "STOP"])
        task = asyncio.create_task(module.handle_client(socket, handler_args(ws_receive_max_messages=2)))
        try:
            await until(entered.is_set)
            await until(lambda: socket.closed, 1)
            assert socket.closed[0][0] == 1013
        finally:
            release.set()
            await asyncio.wait_for(task, 3)
        assert sessions[0].audio == [b"a"]
        assert sessions[0].completed == [b"a"]
        assert sessions[0].previews == []
        assert sessions[0].finals == []
        assert socket.sent == [{"event": "started"}]

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["partial", "commit", "stop-client", "stop-server"])
@pytest.mark.parametrize("failure", ["overflow", "peer-close", "receive-error"])
def test_inflight_result_is_not_success_after_receive_failure(monkeypatch, operation, failure):
    module = load_service_module()
    sessions = session_fixture(monkeypatch, module)
    entered, release = threading.Event(), threading.Event()
    session_type = module.RealtimeASRSession
    method = "commit" if operation in {"commit", "stop-client"} else "decode"
    original = getattr(session_type, method)

    def blocked(self, *args, **kwargs):
        entered.set()
        assert release.wait(8), "owned final/preview worker was not released"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(session_type, method, blocked)
    command = [] if operation == "partial" else ["COMMIT" if operation == "commit" else "STOP"]

    class FailureAfterWork(Socket):
        failed = False

        async def __anext__(self):
            message = await super().__anext__()
            if message == "ABORT-BARRIER":
                await until(entered.is_set)
                if failure == "peer-close":
                    self.close_code = 1000
                    self.failed = True
                    raise StopAsyncIteration
                if failure == "receive-error":
                    self.failed = True
                    raise RuntimeError("synthetic receive error during worker")
                return b"queued"
            return message

    async def exercise():
        socket = FailureAfterWork(["START", b"a", *command, "ABORT-BARRIER",
                                   *([b"queued"] * 8), "STOP"])
        args = handler_args(ws_receive_max_messages=8,
                            decode_interval=0 if operation == "partial" else 1e30,
                            endpoint_mode="server" if operation == "stop-server" else "client")
        task = asyncio.create_task(module.handle_client(socket, args))
        try:
            await until(entered.is_set)
            await until(lambda: socket.failed or socket.closed)
            if failure == "overflow":
                assert socket.closed[0][0] == 1013
        finally:
            release.set()
            await asyncio.wait_for(task, 3)
        assert sessions[0].audio == [b"a"]
        assert sessions[0].completed == [b"a"]
        assert socket.sent == [{"event": "started"}], "failed session emitted successful result/STOP"
        if operation == "partial":
            assert sessions[0].previews == [1]
            assert sessions[0].finals == []
        else:
            assert sessions[0].previews == []
            assert sessions[0].finals == [1]

    asyncio.run(exercise())


async def receive_until_stopped(client):
    values = []
    while True:
        value = json.loads(await asyncio.wait_for(client.recv(), 3))
        values.append(value)
        if value.get("event") == "stopped":
            return values


@pytest.mark.parametrize("overflow", [False, True])
def test_real_websocket_burst_pong_or_explicit_overload(monkeypatch, overflow):
    import websockets
    from websockets.exceptions import ConnectionClosed

    module = load_service_module()
    module.ConnectionClosed = ConnectionClosed
    entered, release = threading.Event(), threading.Event()
    sessions = session_fixture(monkeypatch, module, block="decode", entered=entered, release=release)
    args = handler_args(ws_receive_max_messages=8 if overflow else 128)

    async def exercise():
        handler_tasks = []

        async def handler(socket, path=None):
            task = asyncio.current_task()
            handler_tasks.append(task)
            await module.handle_client(socket, args)

        async with websockets.serve(handler, "127.0.0.1", 0, ping_interval=None,
                                    close_timeout=0.2) as server:
            port = server.sockets[0].getsockname()[1]
            try:
                async with websockets.connect(f"ws://127.0.0.1:{port}", ping_interval=None,
                                              close_timeout=0.2) as client:
                    pong = None
                    try:
                        await client.send("START")
                        assert json.loads(await asyncio.wait_for(client.recv(), 2)) == {"event": "started"}
                        first = b"first" + bytes(6395)
                        await client.send(first)
                        await until(entered.is_set)
                        frames = [i.to_bytes(2, "little") + bytes(6398) for i in range(48)]
                        for frame in frames:
                            try:
                                await client.send(frame)
                            except ConnectionClosed:
                                assert overflow
                                break
                            await asyncio.sleep(0.003)
                        if overflow:
                            with pytest.raises(ConnectionClosed) as caught:
                                await asyncio.wait_for(client.recv(), 1)
                            assert caught.value.rcvd is not None
                            assert caught.value.rcvd.code == 1013
                            release.set()
                            await until(lambda: all(t.done() for t in handler_tasks))
                            assert sessions[0].audio == [first]
                            assert sessions[0].finals == []
                        else:
                            pong = await client.ping(b"receive-pressure-test")
                            await asyncio.wait_for(asyncio.shield(pong), 0.5)
                            assert not release.is_set(), "Pong must arrive while decoder is blocked"
                            await client.send("STOP")
                            release.set()
                            received = await receive_until_stopped(client)
                            assert sessions[0].audio == [first, *frames]
                            assert sessions[0].completed == sessions[0].audio
                            assert sessions[0].finals == [49]
                            assert [r["count"] for r in received if r.get("event") == "final"] == [49]
                    finally:
                        release.set()
                        if pong is not None:
                            await asyncio.wait_for(asyncio.gather(pong, return_exceptions=True), 2)
            finally:
                release.set()
                if handler_tasks:
                    await asyncio.wait_for(asyncio.gather(*handler_tasks, return_exceptions=True), 4)

    asyncio.run(exercise())


def test_real_connections_do_not_share_queue_or_block_each_other(monkeypatch):
    import websockets
    from websockets.exceptions import ConnectionClosed

    module = load_service_module()
    module.ConnectionClosed = ConnectionClosed
    sessions = session_fixture(monkeypatch, module)
    entered, release = threading.Event(), threading.Event()
    original = module.RealtimeASRSession.decode

    def decode(self, is_final=False):
        if self.audio == [b"slow"]:
            entered.set()
            assert release.wait(8)
        return original(self, is_final)

    monkeypatch.setattr(module.RealtimeASRSession, "decode", decode)

    async def exercise():
        handlers = []

        async def handler(socket, path=None):
            handlers.append(asyncio.current_task())
            await module.handle_client(socket, handler_args())

        async with websockets.serve(handler, "127.0.0.1", 0, ping_interval=None,
                                    close_timeout=0.2) as server:
            uri = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            try:
                async with websockets.connect(uri, ping_interval=None, close_timeout=0.2) as slow:
                    try:
                        await slow.send("START")
                        await asyncio.wait_for(slow.recv(), 2)
                        await slow.send(b"slow")
                        await until(entered.is_set)
                        async with websockets.connect(uri, ping_interval=None, close_timeout=0.2) as fast:
                            await fast.send("START")
                            await asyncio.wait_for(fast.recv(), 2)
                            await fast.send(b"fast-a")
                            await fast.send(b"fast-b")
                            await fast.send("STOP")
                            await receive_until_stopped(fast)
                        assert not release.is_set()
                        assert sessions[1].audio == [b"fast-a", b"fast-b"]
                        assert sessions[1].finals == [2]
                        release.set()
                        await slow.send("STOP")
                        await receive_until_stopped(slow)
                        assert sessions[0].audio == [b"slow"]
                        assert sessions[0].finals == [1]
                    finally:
                        release.set()
            finally:
                release.set()
                if handlers:
                    await asyncio.wait_for(asyncio.gather(*handlers, return_exceptions=True), 4)

    asyncio.run(exercise())
