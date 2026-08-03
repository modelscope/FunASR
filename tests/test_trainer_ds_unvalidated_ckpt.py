import os

import pytest
import torch

from funasr.train_utils.trainer import Trainer as TrainerTorch
from funasr.train_utils.trainer_ds import Trainer as TrainerDs


class _DummyModule(torch.nn.Module):
    """A tiny real model for the torch.save checkpoint path."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


class _FakeDeepSpeedEngine:
    """Minimal stand-in for a DeepSpeed engine (only save_checkpoint is used).

    DeepSpeed writes one checkpoint directory per tag; this fake writes a
    placeholder *file* per tag so that keep_nbest_models pruning's
    smart_remove() actually deletes something observable on disk.
    """

    def __init__(self):
        self.tags = []
        self.client_states = []

    def save_checkpoint(self, save_dir=None, tag=None, client_state=None):
        self.tags.append(tag)
        self.client_states.append(client_state)
        with open(os.path.join(save_dir, tag), "w") as f:
            f.write("placeholder")
        return True


# The same ranking logic exists in three checkpointing paths:
#   trainer_ds_torch     -- TrainerDs, use_deepspeed=False (torch.save path)
#   trainer_ds_deepspeed -- TrainerDs, use_deepspeed=True  (DeepSpeed save path)
#   trainer_torch        -- non-DeepSpeed Trainer (funasr-train), torch.save path
SAVE_PATHS = ["trainer_ds_torch", "trainer_ds_deepspeed", "trainer_torch"]


def _make_trainer(tmp_path, ranking, save_path, keep_nbest_models=1):
    kwargs = dict(
        output_dir=str(tmp_path),
        device="cpu",
        keep_nbest_models=keep_nbest_models,
        avg_keep_nbest_models_type=ranking,
    )
    if save_path == "trainer_torch":
        return TrainerTorch(local_rank=0, **kwargs)
    return TrainerDs(
        rank=0,
        local_rank=0,
        world_size=1,
        use_ddp=False,
        use_fsdp=False,
        use_fp16=False,
        use_bf16=False,
        use_deepspeed=(save_path == "trainer_ds_deepspeed"),
        **kwargs,
    )


def _make_fixtures(save_path):
    if save_path == "trainer_ds_deepspeed":
        return _FakeDeepSpeedEngine(), None, None, None
    model = _DummyModule()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1)
    return model, optim, scheduler, None


def _good_metric(ranking):
    # a loss is good when it is low; an acc is good when it is high
    return 0.5 if ranking == "loss" else 0.9


def _bad_metric(ranking):
    return 1.0 if ranking == "loss" else 0.5


@pytest.mark.parametrize("save_path", SAVE_PATHS)
@pytest.mark.parametrize("ranking", ["loss", "acc"])
def test_unvalidated_checkpoint_cannot_evict_validated_best(tmp_path, ranking, save_path):
    """A checkpoint saved at an unvalidated step must not enter ranking.

    Regression for the KeyError fix: it used to fall back to a fabricated
    score of 0.0, which under loss-based ranking pruned the validated best
    checkpoint and left only the unvalidated one.
    """
    trainer = _make_trainer(tmp_path, ranking, save_path, keep_nbest_models=1)
    metric_dict = getattr(trainer, f"val_{ranking}_step_or_epoch")
    model, optim, scheduler, scaler = _make_fixtures(save_path)

    # step 1 was validated; register its metric so it is a ranked best.
    metric_dict["model.pt.ep1.1"] = _bad_metric(ranking)
    trainer.save_checkpoint(
        epoch=1, step=1, step_in_epoch=1,
        model=model, optim=optim, scheduler=scheduler, scaler=scaler,
    )
    assert trainer.saved_ckpts == {"model.pt.ep1.1": _bad_metric(ranking)}
    assert trainer.best_step_or_epoch == "model.pt.ep1.1"

    # step 2 was never validated (e.g. save_checkpoint_interval is not a
    # multiple of validate_interval) -> save_checkpoint used to KeyError here.
    trainer.save_checkpoint(
        epoch=1, step=2, step_in_epoch=2,
        model=model, optim=optim, scheduler=scheduler, scaler=scaler,
    )

    # The unvalidated checkpoint is kept on disk but excluded from ranking:
    # no synthetic score, no best change, and the validated best is never pruned.
    # Both the validated best and the unvalidated checkpoint files survive.
    assert trainer.saved_ckpts == {"model.pt.ep1.1": _bad_metric(ranking)}
    assert trainer.best_step_or_epoch == "model.pt.ep1.1"
    assert os.path.exists(os.path.join(tmp_path, "model.pt.ep1.1"))
    assert os.path.exists(os.path.join(tmp_path, "model.pt.ep1.2"))


@pytest.mark.parametrize("save_path", SAVE_PATHS)
@pytest.mark.parametrize("ranking", ["loss", "acc"])
def test_validated_checkpoints_still_rank_and_prune(tmp_path, ranking, save_path):
    """Validated checkpoints must keep ranking and keep_nbest_models pruning."""
    trainer = _make_trainer(tmp_path, ranking, save_path, keep_nbest_models=1)
    metric_dict = getattr(trainer, f"val_{ranking}_step_or_epoch")
    model, optim, scheduler, scaler = _make_fixtures(save_path)

    metric_dict["model.pt.ep1.1"] = _bad_metric(ranking)
    metric_dict["model.pt.ep1.2"] = _good_metric(ranking)

    trainer.save_checkpoint(
        epoch=1, step=1, step_in_epoch=1,
        model=model, optim=optim, scheduler=scheduler, scaler=scaler,
    )
    trainer.save_checkpoint(
        epoch=1, step=2, step_in_epoch=2,
        model=model, optim=optim, scheduler=scheduler, scaler=scaler,
    )

    assert trainer.saved_ckpts == {"model.pt.ep1.2": _good_metric(ranking)}
    assert trainer.best_step_or_epoch == "model.pt.ep1.2"
    # the worse validated checkpoint was pruned, the better one remains
    assert not os.path.exists(os.path.join(tmp_path, "model.pt.ep1.1"))
    assert os.path.exists(os.path.join(tmp_path, "model.pt.ep1.2"))
