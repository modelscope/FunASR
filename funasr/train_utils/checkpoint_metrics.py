"""Validation metric availability shared by the torch and DeepSpeed trainers."""

import logging
import math

import torch
import torch.distributed as dist


def finite_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class ValidationMetrics:
    def __init__(self):
        self.count = 0
        self.sums = {"loss": 0.0, "acc": 0.0}
        self.complete = {"loss": True, "acc": True}

    def update(self, loss, stats):
        loss = finite_scalar(loss)
        values = {"loss": loss, "acc": finite_scalar(stats.get("acc"))}
        self.count += 1
        for name, value in values.items():
            if value is None or loss is None:
                self.complete[name] = False
            else:
                self.sums[name] += value

    def compute(self, device, distributed=False):
        names = ("loss", "acc")
        means = [self.sums[name] / max(self.count, 1) for name in names]
        available = [self.count > 0 and self.complete[name] for name in names]
        if distributed:
            # All ranks enter both collectives, including empty/nonfinite ranks.
            values = torch.tensor(means, dtype=torch.float32, device=device)
            flags = torch.tensor(available, dtype=torch.int32, device=device)
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            dist.all_reduce(flags, op=dist.ReduceOp.MIN)
            means = (values / dist.get_world_size()).cpu().tolist()
            available = flags.cpu().tolist()
        return {
            name: finite_scalar(value) if valid else None
            for name, value, valid in zip(names, means, available)
        }


def record_validation_metrics(trainer, ckpt_name, metrics):
    selected = trainer.avg_keep_nbest_models_type
    reject = metrics[selected] is None and trainer.best_step_or_epoch == ckpt_name
    distributed = trainer.use_ddp or trainer.use_fsdp or getattr(trainer, "use_deepspeed", False)
    if metrics[selected] is None and distributed:
        # Only rank zero may have updated best during checkpoint saving.
        flag = torch.tensor(int(reject), dtype=torch.int32, device=trainer.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        reject = bool(flag.item())
    if reject:
        raise ValueError(
            f"Cannot overwrite current best checkpoint {ckpt_name} with unavailable "
            f"validation {selected}; use a new checkpoint name or output directory."
        )
    for name, value in metrics.items():
        setattr(trainer, f"val_{name}_avg", value if value is not None else float("nan"))
        history = getattr(trainer, f"val_{name}_step_or_epoch")
        if value is None:
            history.pop(ckpt_name, None)
        else:
            history[ckpt_name] = value
    if metrics[selected] is None:
        trainer.saved_ckpts.pop(ckpt_name, None)
        if trainer.rank == 0 and not trainer._warned_validation_metric:
            remedy = (
                " For models without ASR accuracy, start a new output directory with "
                "++train_conf.avg_keep_nbest_models_type=loss; acc_rich is not ASR accuracy."
                if selected == "acc" else " Check the validation data and model outputs."
            )
            logging.warning(
                "Validation has no complete, finite %s metric on every batch/rank; "
                "this checkpoint is excluded from best, pruning and averaging.%s",
                selected, remedy,
            )
            trainer._warned_validation_metric = True


def check_resume_ranking(checkpoint, configured):
    previous = checkpoint.get("avg_keep_nbest_models_type")
    if previous is not None and previous != configured:
        raise ValueError(
            f"Cannot change checkpoint ranking metric from {previous} to {configured} "
            "while resuming. Use a new output directory to avoid mixing metric histories."
        )
