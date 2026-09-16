"""CPU/Gloo regression tests for data2vec's compute_var (PR #3710).

PR #3710 replaced a hardcoded ``zc = torch.tensor(y.size(0)).cuda()`` with
``zc = torch.tensor(y.size(0), device=y.device)`` so the distributed variance
reduction no longer assumes a CUDA device. These tests pin that contract:
  * the uninitialized (single-process, no process group) branch works on CPU;
  * the initialized branch runs with a real 2-rank Gloo process group and
    unequal local row counts (exercising the all_reduce).
"""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from funasr.models.data2vec.data2vec_encoder import Data2VecEncoder

compute_var = Data2VecEncoder.compute_var


def _rank_inputs(rank):
    # Unequal local row counts across ranks, 3 features each.
    if rank == 0:
        return torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        )
    return torch.tensor(
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
        dtype=torch.float64,
    )


def _worker(rank, world_size, init_uri, results):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_uri,
        timeout=timedelta(seconds=30),
    )
    try:
        out = compute_var(_rank_inputs(rank))
        results[rank] = {"device": str(out.device), "value": float(out)}
    except Exception as exc:  # pragma: no cover - surfaced in the parent assert
        results[rank] = {"error": repr(exc)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _f32_worker(rank, world_size, init_uri, results):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_uri,
        timeout=timedelta(seconds=30),
    )
    try:
        y = _rank_inputs(rank).float()
        out = compute_var(y)
        results[rank] = {"device": str(out.device), "dtype": str(out.dtype)}
    except Exception as exc:  # pragma: no cover - surfaced in the parent assert
        results[rank] = {"error": repr(exc)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_two_rank(tmp_path, world_size=2, tag="default"):
    init_uri = (tmp_path / f"gloo-compute-var-{tag}").as_uri()
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    results = mgr.dict()
    procs = [
        ctx.Process(target=_worker, args=(rank, world_size, init_uri, results))
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited with code {p.exitcode}"
    return dict(results)


def _run_two_rank_with(tmp_path, worker_fn, world_size=2, tag="default"):
    init_uri = (tmp_path / f"gloo-compute-var-{tag}").as_uri()
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    results = mgr.dict()
    procs = [
        ctx.Process(target=worker_fn, args=(rank, world_size, init_uri, results))
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited with code {p.exitcode}"
    return dict(results)


def _reference_var(y):
    # Same as the uninitialized branch in compute_var.
    return torch.sqrt(y.var(dim=0) + 1e-6).mean()


class TestComputeVar:
    """Regression tests for compute_var (PR #3710 — device-agnostic)."""

    def test_uninitialized_cpu(self):
        """No process group: compute_var runs on CPU and matches torch.var."""
        y = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float64,
        )
        result = compute_var(y)
        assert result.device == torch.device("cpu")
        assert result.shape == torch.Size([])
        assert torch.allclose(result, _reference_var(y))
        assert torch.isfinite(result) and result > 0

    def test_two_rank_gloo_unequal_rows(self, tmp_path):
        """Two-rank Gloo, unequal local rows: device-agnostic all_reduce path."""
        results = _run_two_rank(tmp_path, world_size=2)
        assert len(results) == 2
        # Both ranks must end on CPU and agree on the all-reduced variance.
        for rank in (0, 1):
            assert "error" not in results[rank], results[rank]
            assert results[rank]["device"] == "cpu", results[rank]
        assert abs(results[0]["value"] - results[1]["value"]) < 1e-9, results

        # Cross-check against the uninitialized reference on the concatenated
        # data (what all_reduce computes globally).
        y_all = torch.cat([_rank_inputs(0), _rank_inputs(1)], dim=0)
        assert abs(results[0]["value"] - float(_reference_var(y_all))) < 1e-9

    def test_two_rank_gloo_float32(self, tmp_path):
        """Gloo path preserves a float32 input (no dtype drift)."""
        results = _run_two_rank_with(tmp_path, _f32_worker, tag="f32")
        for rank in (0, 1):
            assert "error" not in results[rank], results[rank]
            assert results[rank]["device"] == "cpu"
            assert results[rank]["dtype"] == "torch.float32", results[rank]