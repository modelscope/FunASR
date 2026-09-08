"""Exercise validation, checkpoint files and ranking without acoustic weights."""

import logging
from pathlib import Path

import pytest
import torch

from funasr.train_utils.average_nbest_models import _get_checkpoint_paths, average_checkpoints
from funasr.train_utils.trainer import Trainer as TrainerTorch
from funasr.train_utils.trainer_ds import Trainer as TrainerDs


SAVE_PATHS = ["torch", "ds_torch", "ds_engine"]


class MetricModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))

    def forward(self, value, accuracy="missing"):
        stats = {"loss": value, "acc_rich": value.new_tensor(0.8)}
        if accuracy != "missing":
            stats["acc"] = None if accuracy is None else value.new_tensor(accuracy)
        return value, stats, value.new_tensor(1.0)

    def save_checkpoint(self, save_dir, tag, client_state):
        # Model the engine's save boundary, not distributed DeepSpeed training.
        path = Path(save_dir) / tag
        path.mkdir(exist_ok=True)
        torch.save({**client_state, "state_dict": self.state_dict()}, path / "mp_rank_00_model_states.pt")

    def load_checkpoint(self, save_dir, tag):
        path = Path(save_dir) / tag / "mp_rank_00_model_states.pt"
        state = torch.load(path, weights_only=True)
        self.load_state_dict(state["state_dict"])
        return str(path), state


class Batches:
    def __init__(self, values):
        self.values = values
        self.batch_sampler = self

    def set_epoch(self, epoch):
        pass

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for loss, accuracy in self.values:
            yield {"value": torch.tensor(loss), "accuracy": accuracy}


class Run:
    def __init__(self, path, kind, ranking="acc", keep=2):
        self.path = path
        self.kind = kind
        kwargs = dict(output_dir=str(path), device="cpu", log_interval=100,
                      avg_keep_nbest_models_type=ranking, keep_nbest_models=keep)
        self.trainer = (TrainerTorch(local_rank=0, **kwargs) if kind == "torch" else
                        TrainerDs(rank=0, local_rank=0, world_size=1, use_deepspeed=False, **kwargs))
        self.model = MetricModel()
        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1)

    def validate(self, epoch, values):
        if self.kind == "ds_engine":
            self.trainer.use_deepspeed = False
        self.trainer.validate_epoch(model=self.model, dataloader_val=Batches(values), epoch=epoch)

    def save(self, epoch):
        if self.kind == "ds_engine":
            self.trainer.use_deepspeed = True
        self.trainer.save_checkpoint(epoch=epoch, model=self.model,
                                     optim=self.optim, scheduler=self.scheduler)

    def checkpoint(self, name):
        path = self.path / name
        if self.kind == "ds_engine":
            path = path / "mp_rank_00_model_states.pt"
        return torch.load(path, weights_only=True)

    def close(self):
        writer = getattr(self.trainer, "writer", None)
        if writer is not None:
            writer.close()


@pytest.fixture(params=SAVE_PATHS)
def run(request, tmp_path):
    case = Run(tmp_path, request.param)
    yield case
    case.close()


@pytest.mark.parametrize("accuracy", ["missing", None, float("nan"), float("inf")])
def test_unavailable_accuracy_never_ranks_or_averages(run, accuracy, caplog):
    with caplog.at_level(logging.WARNING):
        for epoch in (1, 2):
            run.validate(epoch, [(1.0, accuracy)])
            run.save(epoch)
    assert run.trainer.val_acc_step_or_epoch == {}
    assert run.trainer.saved_ckpts == {}
    assert run.trainer.best_step_or_epoch == ""
    assert not (run.path / "model.pt.best").exists()
    assert run.checkpoint("model.pt")["val_acc_step_or_epoch"] == {}
    assert _get_checkpoint_paths(str(run.path), use_deepspeed=run.kind == "ds_engine") == []
    messages = [r.message for r in caplog.records if "avg_keep_nbest_models_type=loss" in r.message]
    assert len(messages) == 1


@pytest.mark.parametrize("values", [[(1.0, 0.4), (1.0, "missing")],
                                    [(1.0, "missing"), (1.0, 0.4)]])
def test_partial_accuracy_is_not_a_validation_score(run, values):
    run.validate(1, values)
    run.save(1)
    assert run.trainer.val_acc_step_or_epoch == {}
    assert run.trainer.val_loss_step_or_epoch == {"model.pt.ep1": 1.0}
    assert run.trainer.saved_ckpts == {}


def test_empty_validation_does_not_reuse_previous_averages(run):
    run.validate(1, [(3.0, 0.5)])
    run.save(1)
    run.validate(2, [])
    run.save(2)
    assert "model.pt.ep2" not in run.trainer.val_acc_step_or_epoch
    assert "model.pt.ep2" not in run.trainer.val_loss_step_or_epoch
    assert run.trainer.best_step_or_epoch == "model.pt.ep1"


@pytest.mark.parametrize("bad_loss", [float("nan"), float("inf")])
def test_nonfinite_loss_does_not_publish_partial_metrics(run, bad_loss):
    run.validate(1, [(1.0, 0.4), (bad_loss, 0.5)])
    run.save(1)
    assert run.trainer.val_loss_step_or_epoch == {}
    assert run.trainer.val_acc_step_or_epoch == {}


def test_real_zero_accuracy_preserves_existing_latest_tie(run):
    for epoch in (1, 2, 3):
        with torch.no_grad():
            run.model.marker.fill_(epoch)
        run.validate(epoch, [(1.0, 0.0)])
        run.save(epoch)
    assert run.trainer.best_step_or_epoch == "model.pt.ep3"
    assert run.checkpoint("model.pt.best")["state_dict"]["marker"].item() == 3


def test_validation_logs_current_running_means_not_previous_epoch(run, monkeypatch):
    observed = []
    monkeypatch.setattr(run.trainer, "log", lambda *a, **kw: observed.append(
        (run.trainer.val_loss_avg, run.trainer.val_acc_avg)))
    run.validate(1, [(1.0, 0.2), (3.0, 0.4)])
    run.validate(2, [(5.0, 0.6)])
    assert [v[0] for v in observed] == pytest.approx([1.0, 2.0, 5.0])
    assert [v[1] for v in observed] == pytest.approx([0.2, 0.3, 0.6])


def test_explicit_loss_best_pruning_and_average_agree(run, caplog):
    run.trainer.avg_keep_nbest_models_type = "loss"
    for epoch, loss in enumerate((3.0, 1.0, 2.0), 1):
        with torch.no_grad():
            run.model.marker.fill_(epoch)
        run.validate(epoch, [(loss, "missing")])
        run.save(epoch)
    assert run.trainer.best_step_or_epoch == "model.pt.ep2"
    assert run.checkpoint("model.pt.best")["state_dict"]["marker"].item() == 2
    assert set(run.trainer.saved_ckpts) == {"model.pt.ep2", "model.pt.ep3"}
    path = average_checkpoints(str(run.path), last_n=2, use_deepspeed=run.kind == "ds_engine")
    assert torch.load(path, weights_only=True)["state_dict"]["marker"].item() == 2.5
    assert not any("avg_keep_nbest_models_type=loss" in r.message for r in caplog.records)


def test_revalidation_removes_stale_nonbest_metric_before_serialization(run):
    run.validate(1, [(1.0, 0.9)])
    run.save(1)
    run.validate(2, [(1.0, 0.4)])
    run.save(2)
    run.validate(2, [(1.0, "missing")])
    run.save(2)
    state = run.checkpoint("model.pt")
    assert "model.pt.ep2" not in state["val_acc_step_or_epoch"]
    assert "model.pt.ep2" not in state["saved_ckpts"]
    assert run.trainer.best_step_or_epoch == "model.pt.ep1"


def test_invalid_revalidation_cannot_overwrite_current_best(run):
    run.validate(1, [(1.0, 0.5)])
    run.save(1)
    before = run.checkpoint("model.pt.best")
    with pytest.raises(ValueError, match="current best"):
        run.validate(1, [(1.0, "missing")])
    assert run.trainer.val_acc_step_or_epoch == {"model.pt.ep1": 0.5}
    assert torch.equal(before["state_dict"]["marker"],
                       run.checkpoint("model.pt.best")["state_dict"]["marker"])


def test_resume_preserves_unavailable_metric_exclusion(run):
    run.validate(1, [(1.0, "missing")])
    run.save(1)
    resumed = Run(run.path, run.kind)
    try:
        resumed.trainer.use_deepspeed = run.kind == "ds_engine" if run.kind != "torch" else False
        resumed.trainer.resume_checkpoint(model=resumed.model, optim=resumed.optim, scheduler=resumed.scheduler)
        assert resumed.trainer.val_acc_step_or_epoch == {}
        assert resumed.trainer.saved_ckpts == {}
    finally:
        resumed.close()


def test_resume_cannot_mix_accuracy_history_with_loss_ranking(run):
    run.validate(1, [(1.0, 0.0)])
    run.save(1)
    resumed = Run(run.path, run.kind, ranking="loss")
    try:
        if run.kind != "torch":
            resumed.trainer.use_deepspeed = run.kind == "ds_engine"
        with pytest.raises(ValueError, match="ranking metric"):
            resumed.trainer.resume_checkpoint(model=resumed.model, optim=resumed.optim, scheduler=resumed.scheduler)
    finally:
        resumed.close()


def test_resume_retains_current_best_guard(run):
    run.validate(1, [(1.0, 0.5)])
    run.save(1)
    resumed = Run(run.path, run.kind)
    try:
        if run.kind != "torch":
            resumed.trainer.use_deepspeed = run.kind == "ds_engine"
        resumed.trainer.resume_checkpoint(model=resumed.model, optim=resumed.optim, scheduler=resumed.scheduler)
        assert resumed.trainer.best_step_or_epoch == "model.pt.ep1"
        with pytest.raises(ValueError, match="current best"):
            resumed.validate(1, [(1.0, "missing")])
    finally:
        resumed.close()


@pytest.mark.parametrize("ranking,values", [("acc", [0.3, 0.9, 0.5]), ("loss", [3.0, 1.0, 2.0])])
def test_persisted_ranking_matches_best_and_pruned_files(run, ranking, values):
    run.trainer.avg_keep_nbest_models_type = ranking
    run.trainer.keep_nbest_models = 1
    for epoch, value in enumerate(values, 1):
        loss, acc = (1.0, value) if ranking == "acc" else (value, "missing")
        run.validate(epoch, [(loss, acc)])
        run.save(epoch)
        state = run.checkpoint("model.pt")
        assert state["best_step_or_epoch"] == run.trainer.best_step_or_epoch
        assert state["saved_ckpts"] == run.trainer.saved_ckpts
        best = run.checkpoint("model.pt.best")
        assert best["best_step_or_epoch"] == run.trainer.best_step_or_epoch
        assert all((run.path / name).exists() for name in state["saved_ckpts"])
    assert not (run.path / "model.pt.ep3").exists()
    assert (run.path / "model.pt").exists()


@pytest.mark.parametrize("failed_tag", ["model.pt", "model.pt.best"])
def test_write_failure_does_not_delete_previous_candidate(run, failed_tag, monkeypatch):
    run.trainer.keep_nbest_models = 1
    with torch.no_grad():
        run.model.marker.fill_(1)
    run.validate(1, [(1.0, 0.3)])
    run.save(1)
    with torch.no_grad():
        run.model.marker.fill_(2)
    run.validate(2, [(1.0, 0.9)])
    if run.kind == "ds_engine":
        original = run.model.save_checkpoint

        def fail_engine(save_dir, tag, client_state):
            if tag == failed_tag:
                raise OSError("simulated checkpoint write failure")
            return original(save_dir, tag, client_state)

        monkeypatch.setattr(run.model, "save_checkpoint", fail_engine)
    else:
        original = torch.save

        def fail_torch(state, path, *args, **kwargs):
            if Path(path).name == failed_tag:
                raise OSError("simulated checkpoint write failure")
            return original(state, path, *args, **kwargs)

        monkeypatch.setattr(torch, "save", fail_torch)
    with pytest.raises(OSError, match="simulated checkpoint"):
        run.save(2)
    assert (run.path / "model.pt.ep1").exists()
    assert run.checkpoint("model.pt.best")["state_dict"]["marker"].item() == 1


@pytest.mark.parametrize("ranking,scores", [("acc", {"model.pt.ep1": 0.9, "model.pt.ep2": 0.8, "model.pt.ep3": float("nan")}),
                                          ("loss", {"model.pt.ep1": 0.1, "model.pt.ep2": 0.2, "model.pt.ep3": float("nan")})])
def test_average_filters_nonfinite_and_missing_files_before_taking_n(tmp_path, ranking, scores):
    torch.save({"avg_keep_nbest_models_type": ranking, f"val_{ranking}_step_or_epoch": scores}, tmp_path / "model.pt")
    for epoch in (2, 3):
        torch.save({"state_dict": {"x": torch.tensor(float(epoch))}}, tmp_path / f"model.pt.ep{epoch}")
    assert _get_checkpoint_paths(str(tmp_path), last_n=1) == [str(tmp_path / "model.pt.ep2")]
