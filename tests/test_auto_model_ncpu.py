"""Regression cover for applying `ncpu` to BLAS as well as to torch.

Both settings are process-wide and follow `torch.set_num_threads` semantics.
The helper is driven directly for the update rule, and through the real
`AutoModel.build_model` path for the interaction between them -- swapping the
BLAS limiter touches torch's OpenMP pool as a side effect, which only shows up
across a sequence of builds.
"""

import numpy as np
import pytest
import scipy.linalg
import torch
import threadpoolctl

from funasr.auto.auto_model import (
    _current_blas_threads,
    _limit_blas_threads,
)


@pytest.fixture(autouse=True)
def _restore_thread_settings():
    """Leave both thread settings where the test found them.

    The BLAS limit is deliberately process-wide and never released, so without
    this a test would leak its thread count into every later test -- and the
    torch count needs restoring for the same reason, since tests here set it
    to make their assertions readable.
    """
    original_blas = _current_blas_threads()
    original_torch = torch.get_num_threads()
    yield
    if original_blas is not None:
        _limit_blas_threads(original_blas)
    torch.set_num_threads(original_torch)


def _require_blas_pools():
    if _current_blas_threads() is None:
        pytest.skip("no BLAS pool is loaded in this environment")


class _ProbeModel(torch.nn.Module):
    """Smallest thing `build_model` accepts, so no weights are downloaded."""

    def __init__(self, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


@pytest.fixture
def probe_model():
    from funasr.register import tables

    name = "ncpu-blas-probe"
    added = name not in tables.model_classes
    if added:
        tables.model_classes[name] = _ProbeModel
    yield name
    if added:
        tables.model_classes.pop(name, None)


def _build(probe_model, ncpu):
    from funasr.auto.auto_model import AutoModel

    AutoModel(
        model=probe_model,
        model_conf={},
        device="cpu",
        ncpu=ncpu,
        disable_update=True,
    )
    return torch.get_num_threads(), _current_blas_threads()


def test_build_model_sets_both_settings_in_both_orders(probe_model):
    """End-to-end through `build_model`, where the settings are applied.

    Replacing the BLAS limiter drops the previous `threadpool_limits`, and that
    object restores the OpenMP pools as well -- so without care the swap rolls
    torch's thread count back to its value at the previous build, undoing the
    `torch.set_num_threads(ncpu)` on the line above. Asserting only on the
    helper misses this; a sequence of builds is what surfaces it.
    """
    _require_blas_pools()

    for order in ((8, 1), (1, 8), (4, 2, 7)):
        for ncpu in order:
            torch_threads, blas_threads = _build(probe_model, ncpu)
            assert blas_threads == ncpu, (
                "order %s: BLAS settled on %s at ncpu=%s"
                % (order, blas_threads, ncpu)
            )
            assert torch_threads == ncpu, (
                "order %s: torch settled on %s at ncpu=%s"
                % (order, torch_threads, ncpu)
            )


def test_build_model_without_speaker_clustering(probe_model):
    """`ncpu` reaches BLAS whether or not clustering is configured."""
    _require_blas_pools()

    torch_threads, blas_threads = _build(probe_model, 3)
    assert blas_threads == 3
    assert torch_threads == 3


def test_latest_ncpu_wins():
    """Later construction must be able to lower the limit.

    The first revision returned early once a limit existed, so ncpu=1 after
    ncpu=8 left BLAS at 8 -- exactly the case this change exists to fix.
    """
    _require_blas_pools()

    _limit_blas_threads(8)
    assert _current_blas_threads() == 8

    _limit_blas_threads(1)
    assert _current_blas_threads() == 1, (
        "a later, lower ncpu must take effect"
    )


def test_latest_ncpu_wins_in_both_orders():
    _require_blas_pools()

    for first, second in ((8, 1), (1, 8), (4, 2), (2, 4)):
        _limit_blas_threads(first)
        assert _current_blas_threads() == first
        _limit_blas_threads(second)
        assert _current_blas_threads() == second, (
            "order %s -> %s did not settle on the newest value"
            % (first, second)
        )


def test_repeated_same_value_is_stable():
    """Re-setting the current value must be a no-op, not a nested limit."""
    _require_blas_pools()

    _limit_blas_threads(3)
    for _ in range(4):
        _limit_blas_threads(3)
        assert _current_blas_threads() == 3


def test_limit_is_process_wide_and_cross_thread():
    """BLAS pools are process-global, so a worker thread sees the cap too.

    This is the property that makes the fix work at all: the clustering call
    runs on a request worker, not on the thread that built the model.
    """
    _require_blas_pools()

    import threading

    _limit_blas_threads(2)
    seen = []

    def worker():
        seen.append(
            sorted(
                {
                    pool["num_threads"]
                    for pool in threadpoolctl.threadpool_info()
                    if pool["user_api"] == "blas"
                }
            )
        )

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert seen == [[2]], "a worker thread did not observe the cap: %s" % seen


def test_the_cap_actually_reduces_blas_work():
    """The point of the change: fewer threads must mean less CPU.

    A 357-row Laplacian is the size a 35-minute recording produces.
    """
    _require_blas_pools()

    import os

    clock = os.sysconf("SC_CLK_TCK")

    def process_cpu_seconds():
        fields = open("/proc/self/stat").read().split()
        return (int(fields[13]) + int(fields[14])) / clock

    rng = np.random.RandomState(0)
    laplacian = rng.rand(357, 357)
    laplacian = laplacian + laplacian.T

    def measure():
        before = process_cpu_seconds()
        scipy.linalg.eigh(laplacian)
        return process_cpu_seconds() - before

    threads = os.cpu_count() or 1
    if threads < 4:
        pytest.skip("needs at least 4 cores to show a difference")

    _limit_blas_threads(1)
    one_thread_cpu = measure()
    _limit_blas_threads(threads)
    many_thread_cpu = measure()

    assert many_thread_cpu > one_thread_cpu, (
        "capping BLAS did not reduce CPU: %d threads=%.3f core-s, "
        "1 thread=%.3f core-s" % (threads, many_thread_cpu, one_thread_cpu)
    )
