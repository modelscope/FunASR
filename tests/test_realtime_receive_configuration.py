import asyncio
import threading

import pytest

from test_realtime_ws_service import load_service_module


def test_receive_buffer_defaults_are_finite():
    args = load_service_module().build_arg_parser().parse_args([])
    assert getattr(args, "ws_receive_max_messages", None) == 128
    assert getattr(args, "ws_receive_max_bytes", None) == 16 * 1024 * 1024


@pytest.mark.parametrize("flag", ["--ws-receive-max-messages", "--ws-receive-max-bytes"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_receive_buffer_limits_reject_nonpositive_values(flag, value):
    parser = load_service_module().build_arg_parser()
    # A supported positive value must work before testing rejection.
    assert vars(parser.parse_args([flag, "1"]))[flag[2:].replace("-", "_")] == 1
    with pytest.raises(SystemExit):
        parser.parse_args([flag, value])


def test_cancelling_session_work_waits_for_the_owned_worker():
    module = load_service_module()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work():
        entered.set()
        assert release.wait(3)
        finished.set()

    async def exercise():
        task = asyncio.create_task(module.run_session_work(None, work))
        try:
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done(), "cancellation abandoned a still-mutating worker"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())
