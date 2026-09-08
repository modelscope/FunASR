"""Actual two-rank CPU collectives; no distributed acoustic training claim."""

from datetime import timedelta
import json
from pathlib import Path
import time

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from test_checkpoint_metric_presence import Run


def _validate_worker(rank, root):
    root = Path(root)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=(root / "gloo-init").as_uri(),
                            timeout=timedelta(seconds=30))
    results = {}
    try:
        for kind in ("torch", "ds_torch", "ds_engine"):
            for case in ("valid", "missing", "nonfinite", "empty", "reject_best"):
                run = Run(root / f"{kind}-{case}-{rank}", kind)
                run.trainer.rank = rank
                run.trainer.world_size = 2
                run.trainer.use_ddp = True
                values = ([(1.0, 0.2), (3.0, 0.4)] if rank == 0 else
                          [(3.0, 0.6), (5.0, 0.8)])
                if rank == 1 and case == "missing":
                    values[-1] = (5.0, "missing")
                if rank == 1 and case == "nonfinite":
                    values[-1] = (float("nan"), 0.8)
                if rank == 1 and case == "empty":
                    values = []
                try:
                    run.validate(1, values)
                    if case == "reject_best":
                        run.save(1)
                        if kind == "ds_engine":
                            assert run.trainer.saved_ckpts == {"model.pt.ep1": pytest.approx(0.5)}
                            assert run.checkpoint("model.pt")["saved_ckpts"] == run.trainer.saved_ckpts
                        before = dict(run.trainer.val_acc_step_or_epoch)
                        rejected = False
                        try:
                            run.validate(1, [(1.0, "missing")])
                        except ValueError as error:
                            rejected = "current best" in str(error)
                        results[f"{kind}-reject_best"] = {
                            "rejected": rejected,
                            "unchanged": before == run.trainer.val_acc_step_or_epoch,
                        }
                        continue
                    results[f"{kind}-{case}"] = {
                        "acc": run.trainer.val_acc_step_or_epoch,
                        "loss": run.trainer.val_loss_step_or_epoch,
                    }
                finally:
                    run.close()
        (root / f"rank-{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_distributed_validation_agrees_on_availability_and_valid_means(tmp_path):
    assert dist.is_gloo_available(), "CPU Gloo is required for this regression"
    context = mp.spawn(_validate_worker, args=(str(tmp_path),), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 150
        while not context.join(timeout=5):
            assert time.monotonic() < deadline, "Validation collectives did not terminate"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
    assert reports[0] == reports[1]
    for kind in ("torch", "ds_torch", "ds_engine"):
        assert reports[0][f"{kind}-reject_best"] == {"rejected": True, "unchanged": True}
        assert reports[0][f"{kind}-valid"]["acc"]["model.pt.ep1"] == pytest.approx(0.5)
        assert reports[0][f"{kind}-valid"]["loss"]["model.pt.ep1"] == pytest.approx(3.0)
        assert reports[0][f"{kind}-missing"]["acc"] == {}
        assert reports[0][f"{kind}-missing"]["loss"]["model.pt.ep1"] == pytest.approx(3.0)
        for case in ("nonfinite", "empty"):
            assert reports[0][f"{kind}-{case}"] == {"acc": {}, "loss": {}}
