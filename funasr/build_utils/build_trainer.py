# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Trainer module."""
import argparse
import dataclasses
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import is_dataclass
from distutils.version import LooseVersion
from io import BytesIO
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import humanfriendly
import oss2
import torch
import torch.nn
import torch.optim
from typeguard import check_argument_types

from funasr.iterators.abs_iter_factory import AbsIterFactory
from funasr.main_funcs.average_nbest_models import average_nbest_models
from funasr.models.base_model import FunASRModel
from funasr.schedulers.abs_scheduler import AbsBatchStepScheduler
from funasr.schedulers.abs_scheduler import AbsEpochStepScheduler
from funasr.schedulers.abs_scheduler import AbsScheduler
from funasr.schedulers.abs_scheduler import AbsValEpochStepScheduler
from funasr.torch_utils.add_gradient_noise import add_gradient_noise
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.recursive_op import recursive_average
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.train.distributed_utils import DistributedOption
from funasr.train.reporter import Reporter
from funasr.train.reporter import SubReporter
from funasr.utils.build_dataclass import build_dataclass

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
    from torch.cuda.amp import GradScaler
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


    GradScaler = None

try:
    import fairscale
except ImportError:
    fairscale = None


@dataclasses.dataclass
class TrainerOptions:
    ngpu: int
    resume: bool
    use_amp: bool
    train_dtype: str
    grad_noise: bool
    accum_grad: int
    grad_clip: float
    grad_clip_type: float
    log_interval: Optional[int]
    no_forward_run: bool
    use_tensorboard: bool
    use_wandb: bool
    output_dir: Union[Path, str]
    max_epoch: int
    max_update: int
    seed: int
    sharded_ddp: bool
    patience: Optional[int]
    keep_nbest_models: Union[int, List[int]]
    nbest_averaging_interval: int
    early_stopping_criterion: Sequence[str]
    best_model_criterion: Sequence[Sequence[str]]
    val_scheduler_criterion: Sequence[str]
    unused_parameters: bool
    wandb_model_log_interval: int
    use_pai: bool
    oss_bucket: Union[oss2.Bucket, None]


class Trainer:
    """Trainer

    """

    def __init__(self,
                 args,
                 model: FunASRModel,
                 optimizers: Sequence[torch.optim.Optimizer],
                 schedulers: Sequence[Optional[AbsScheduler]],
                 train_dataloader: AbsIterFactory,
                 valid_dataloader: AbsIterFactory,
                 distributed_option: DistributedOption):
        self.trainer_options = self.build_options(args)
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.distributed_option = distributed_option

    def build_options(self, args: argparse.Namespace) -> TrainerOptions:
        """Build options consumed by train(), eval()"""
        assert check_argument_types()
        return build_dataclass(TrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Reserved for future development of another Trainer"""
        pass

    def resume(self,
               checkpoint: Union[str, Path],
               model: torch.nn.Module,
               reporter: Reporter,
               optimizers: Sequence[torch.optim.Optimizer],
               schedulers: Sequence[Optional[AbsScheduler]],
               scaler: Optional[GradScaler],
               ngpu: int = 0,
               ):
        states = torch.load(
            checkpoint,
            map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
        )
        model.load_state_dict(states["model"])
        reporter.load_state_dict(states["reporter"])
        for optimizer, state in zip(optimizers, states["optimizers"]):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, states["schedulers"]):
            if scheduler is not None:
                scheduler.load_state_dict(state)
        if scaler is not None:
            if states["scaler"] is None:
                logging.warning("scaler state is not found")
            else:
                scaler.load_state_dict(states["scaler"])

        logging.info(f"The training was resumed using {checkpoint}")

    def run(self) -> None:
        """Perform training. This method performs the main process of training."""
        assert check_argument_types()
        # NOTE(kamo): Don't check the type more strictly as far trainer_options
        model = self.model
        optimizers = self.optimizers
        schedulers = self.schedulers
        train_dataloader = self.train_dataloader
        valid_dataloader = self.valid_dataloader
        trainer_options = self.trainer_options
        distributed_option = self.distributed_option
        assert is_dataclass(trainer_options), type(trainer_options)
        assert len(optimizers) == len(schedulers), (len(optimizers), len(schedulers))

        if isinstance(trainer_options.keep_nbest_models, int):
            keep_nbest_models = [trainer_options.keep_nbest_models]
        else:
            if len(trainer_options.keep_nbest_models) == 0:
                logging.warning("No keep_nbest_models is given. Change to [1]")
                trainer_options.keep_nbest_models = [1]
            keep_nbest_models = trainer_options.keep_nbest_models

        output_dir = Path(trainer_options.output_dir)
        reporter = Reporter()
        if trainer_options.use_amp:
            if LooseVersion(torch.__version__) < LooseVersion("1.6.0"):
                raise RuntimeError(
                    "Require torch>=1.6.0 for  Automatic Mixed Precision"
                )
            if trainer_options.sharded_ddp:
                if fairscale is None:
                    raise RuntimeError(
                        "Requiring fairscale. Do 'pip install fairscale'"
                    )
                scaler = fairscale.optim.grad_scaler.ShardedGradScaler()
            else:
                scaler = GradScaler()
        else:
            scaler = None

        if trainer_options.resume and (output_dir / "checkpoint.pb").exists():
            self.resume(
                checkpoint=output_dir / "checkpoint.pb",
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                reporter=reporter,
                scaler=scaler,
                ngpu=trainer_options.ngpu,
            )

        start_epoch = reporter.get_epoch() + 1
        if start_epoch == trainer_options.max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )

        if distributed_option.distributed:
            dp_model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=trainer_options.unused_parameters)
        elif distributed_option.ngpu > 1:
            dp_model = torch.nn.parallel.DataParallel(
                model,
                device_ids=list(range(distributed_option.ngpu)),
            )
        else:
            # NOTE(kamo): DataParallel also should work with ngpu=1,
            # but for debuggability it's better to keep this block.
            dp_model = model

        if trainer_options.use_tensorboard and (
                not distributed_option.distributed or distributed_option.dist_rank == 0
        ):
            from torch.utils.tensorboard import SummaryWriter
            if trainer_options.use_pai:
                train_summary_writer = SummaryWriter(
                    os.path.join(trainer_options.output_dir, "tensorboard/train")
                )
                valid_summary_writer = SummaryWriter(
                    os.path.join(trainer_options.output_dir, "tensorboard/valid")
                )
            else:
                train_summary_writer = SummaryWriter(
                    str(output_dir / "tensorboard" / "train")
                )
                valid_summary_writer = SummaryWriter(
                    str(output_dir / "tensorboard" / "valid")
                )
        else:
            train_summary_writer = None

        start_time = time.perf_counter()
        for iepoch in range(start_epoch, trainer_options.max_epoch + 1):
            if iepoch != start_epoch:
                logging.info(
                    "{}/{}epoch started. Estimated time to finish: {}".format(
                        iepoch,
                        trainer_options.max_epoch,
                        humanfriendly.format_timespan(
                            (time.perf_counter() - start_time)
                            / (iepoch - start_epoch)
                            * (trainer_options.max_epoch - iepoch + 1)
                        ),
                    )
                )
            else:
                logging.info(f"{iepoch}/{trainer_options.max_epoch}epoch started")
            set_all_random_seed(trainer_options.seed + iepoch)

            reporter.set_epoch(iepoch)
            # 1. Train and validation for one-epoch
            with reporter.observe("train") as sub_reporter:
                all_steps_are_invalid, max_update_stop = self.train_one_epoch(
                    model=dp_model,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    iterator=train_dataloader.build_iter(iepoch),
                    reporter=sub_reporter,
                    scaler=scaler,
                    summary_writer=train_summary_writer,
                    options=trainer_options,
                    distributed_option=distributed_option,
                )

            with reporter.observe("valid") as sub_reporter:
                self.validate_one_epoch(
                    model=dp_model,
                    iterator=valid_dataloader.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                    distributed_option=distributed_option,
                )

            # 2. LR Scheduler step
            for scheduler in schedulers:
                if isinstance(scheduler, AbsValEpochStepScheduler):
                    scheduler.step(
                        reporter.get_value(*trainer_options.val_scheduler_criterion)
                    )
                elif isinstance(scheduler, AbsEpochStepScheduler):
                    scheduler.step()
            if trainer_options.sharded_ddp:
                for optimizer in optimizers:
                    if isinstance(optimizer, fairscale.optim.oss.OSS):
                        optimizer.consolidate_state_dict()

            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                # 3. Report the results
                logging.info(reporter.log_message())
                if train_summary_writer is not None:
                    reporter.tensorboard_add_scalar(train_summary_writer, key1="train")
                    reporter.tensorboard_add_scalar(valid_summary_writer, key1="valid")
                if trainer_options.use_wandb:
                    reporter.wandb_log()

                # save tensorboard on oss
                if trainer_options.use_pai and train_summary_writer is not None:
                    def write_tensorboard_summary(summary_writer_path, oss_bucket):
                        file_list = []
                        for root, dirs, files in os.walk(summary_writer_path, topdown=False):
                            for name in files:
                                file_full_path = os.path.join(root, name)
                                file_list.append(file_full_path)

                        for file_full_path in file_list:
                            with open(file_full_path, "rb") as f:
                                oss_bucket.put_object(file_full_path, f)

                    write_tensorboard_summary(os.path.join(trainer_options.output_dir, "tensorboard/train"),
                                              trainer_options.oss_bucket)
                    write_tensorboard_summary(os.path.join(trainer_options.output_dir, "tensorboard/valid"),
                                              trainer_options.oss_bucket)

                # 4. Save/Update the checkpoint
                if trainer_options.use_pai:
                    buffer = BytesIO()
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "reporter": reporter.state_dict(),
                            "optimizers": [o.state_dict() for o in optimizers],
                            "schedulers": [
                                s.state_dict() if s is not None else None
                                for s in schedulers
                            ],
                            "scaler": scaler.state_dict() if scaler is not None else None,
                            "ema_model": model.encoder.ema.model.state_dict()
                            if hasattr(model.encoder, "ema") and model.encoder.ema is not None else None,
                        },
                        buffer,
                    )
                    trainer_options.oss_bucket.put_object(os.path.join(trainer_options.output_dir, "checkpoint.pb"),
                                                          buffer.getvalue())
                else:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "reporter": reporter.state_dict(),
                            "optimizers": [o.state_dict() for o in optimizers],
                            "schedulers": [
                                s.state_dict() if s is not None else None
                                for s in schedulers
                            ],
                            "scaler": scaler.state_dict() if scaler is not None else None,
                        },
                        output_dir / "checkpoint.pb",
                    )

                # 5. Save and log the model and update the link to the best model
                if trainer_options.use_pai:
                    buffer = BytesIO()
                    torch.save(model.state_dict(), buffer)
                    trainer_options.oss_bucket.put_object(os.path.join(trainer_options.output_dir,
                                                                       f"{iepoch}epoch.pb"), buffer.getvalue())
                else:
                    torch.save(model.state_dict(), output_dir / f"{iepoch}epoch.pb")

                # Creates a sym link latest.pb -> {iepoch}epoch.pb
                if trainer_options.use_pai:
                    p = os.path.join(trainer_options.output_dir, "latest.pb")
                    if trainer_options.oss_bucket.object_exists(p):
                        trainer_options.oss_bucket.delete_object(p)
                    trainer_options.oss_bucket.copy_object(trainer_options.oss_bucket.bucket_name,
                                                           os.path.join(trainer_options.output_dir,
                                                                        f"{iepoch}epoch.pb"), p)
                else:
                    p = output_dir / "latest.pb"
                    if p.is_symlink() or p.exists():
                        p.unlink()
                    p.symlink_to(f"{iepoch}epoch.pb")

                _improved = []
                for _phase, k, _mode in trainer_options.best_model_criterion:
                    # e.g. _phase, k, _mode = "train", "loss", "min"
                    if reporter.has(_phase, k):
                        best_epoch = reporter.get_best_epoch(_phase, k, _mode)
                        # Creates sym links if it's the best result
                        if best_epoch == iepoch:
                            if trainer_options.use_pai:
                                p = os.path.join(trainer_options.output_dir, f"{_phase}.{k}.best.pb")
                                if trainer_options.oss_bucket.object_exists(p):
                                    trainer_options.oss_bucket.delete_object(p)
                                trainer_options.oss_bucket.copy_object(trainer_options.oss_bucket.bucket_name,
                                                                       os.path.join(trainer_options.output_dir,
                                                                                    f"{iepoch}epoch.pb"), p)
                            else:
                                p = output_dir / f"{_phase}.{k}.best.pb"
                                if p.is_symlink() or p.exists():
                                    p.unlink()
                                p.symlink_to(f"{iepoch}epoch.pb")
                            _improved.append(f"{_phase}.{k}")
                if len(_improved) == 0:
                    logging.info("There are no improvements in this epoch")
                else:
                    logging.info(
                        "The best model has been updated: " + ", ".join(_improved)
                    )

                log_model = (
                        trainer_options.wandb_model_log_interval > 0
                        and iepoch % trainer_options.wandb_model_log_interval == 0
                )
                if log_model and trainer_options.use_wandb:
                    import wandb

                    logging.info("Logging Model on this epoch :::::")
                    artifact = wandb.Artifact(
                        name=f"model_{wandb.run.id}",
                        type="model",
                        metadata={"improved": _improved},
                    )
                    artifact.add_file(str(output_dir / f"{iepoch}epoch.pb"))
                    aliases = [
                        f"epoch-{iepoch}",
                        "best" if best_epoch == iepoch else "",
                    ]
                    wandb.log_artifact(artifact, aliases=aliases)

                # 6. Remove the model files excluding n-best epoch and latest epoch
                _removed = []
                # Get the union set of the n-best among multiple criterion
                nbests = set().union(
                    *[
                        set(reporter.sort_epochs(ph, k, m)[: max(keep_nbest_models)])
                        for ph, k, m in trainer_options.best_model_criterion
                        if reporter.has(ph, k)
                    ]
                )

                # Generated n-best averaged model
                if (
                        trainer_options.nbest_averaging_interval > 0
                        and iepoch % trainer_options.nbest_averaging_interval == 0
                ):
                    average_nbest_models(
                        reporter=reporter,
                        output_dir=output_dir,
                        best_model_criterion=trainer_options.best_model_criterion,
                        nbest=keep_nbest_models,
                        suffix=f"till{iepoch}epoch",
                        oss_bucket=trainer_options.oss_bucket,
                        pai_output_dir=trainer_options.output_dir,
                    )

                for e in range(1, iepoch):
                    if trainer_options.use_pai:
                        p = os.path.join(trainer_options.output_dir, f"{e}epoch.pb")
                        if trainer_options.oss_bucket.object_exists(p) and e not in nbests:
                            trainer_options.oss_bucket.delete_object(p)
                            _removed.append(str(p))
                    else:
                        p = output_dir / f"{e}epoch.pb"
                        if p.exists() and e not in nbests:
                            p.unlink()
                            _removed.append(str(p))
                if len(_removed) != 0:
                    logging.info("The model files were removed: " + ", ".join(_removed))

            # 7. If any updating haven't happened, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    f"The gradients at all steps are invalid in this epoch. "
                    f"Something seems wrong. This training was stopped at {iepoch}epoch"
                )
                break

            if max_update_stop:
                logging.info(
                    f"Stopping training due to "
                    f"num_updates: {trainer_options.num_updates} >= max_update: {trainer_options.max_update}"
                )
                break

            # 8. Check early stopping
            if trainer_options.patience is not None:
                if reporter.check_early_stopping(
                        trainer_options.patience, *trainer_options.early_stopping_criterion
                ):
                    break

        else:
            logging.info(
                f"The training was finished at {trainer_options.max_epoch} epochs "
            )

        # Generated n-best averaged model
        if not distributed_option.distributed or distributed_option.dist_rank == 0:
            average_nbest_models(
                reporter=reporter,
                output_dir=output_dir,
                best_model_criterion=trainer_options.best_model_criterion,
                nbest=keep_nbest_models,
                oss_bucket=trainer_options.oss_bucket,
                pai_output_dir=trainer_options.output_dir,
            )

    def train_one_epoch(
            self,
            model: torch.nn.Module,
            iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
            optimizers: Sequence[torch.optim.Optimizer],
            schedulers: Sequence[Optional[AbsScheduler]],
            scaler: Optional[GradScaler],
            reporter: SubReporter,
            summary_writer,
            options: TrainerOptions,
            distributed_option: DistributedOption,
    ) -> Tuple[bool, bool]:
        assert check_argument_types()

        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        distributed = distributed_option.distributed

        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        all_steps_are_invalid = True
        max_update_stop = False
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        for iiter, (_, batch) in enumerate(
                reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch, dict), type(batch)

            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                all_steps_are_invalid = False
                continue

            with autocast(scaler is not None):
                with reporter.measure_time("forward_time"):
                    retval = model(**batch)

                    # Note(kamo):
                    # Supporting two patterns for the returned value from the model
                    #   a. dict type
                    if isinstance(retval, dict):
                        loss = retval["loss"]
                        stats = retval["stats"]
                        weight = retval["weight"]
                        optim_idx = retval.get("optim_idx")
                        if optim_idx is not None and not isinstance(optim_idx, int):
                            if not isinstance(optim_idx, torch.Tensor):
                                raise RuntimeError(
                                    "optim_idx must be int or 1dim torch.Tensor, "
                                    f"but got {type(optim_idx)}"
                                )
                            if optim_idx.dim() >= 2:
                                raise RuntimeError(
                                    "optim_idx must be int or 1dim torch.Tensor, "
                                    f"but got {optim_idx.dim()}dim tensor"
                                )
                            if optim_idx.dim() == 1:
                                for v in optim_idx:
                                    if v != optim_idx[0]:
                                        raise RuntimeError(
                                            "optim_idx must be 1dim tensor "
                                            "having same values for all entries"
                                        )
                                optim_idx = optim_idx[0].item()
                            else:
                                optim_idx = optim_idx.item()

                    #   b. tuple or list type
                    else:
                        loss, stats, weight = retval
                        optim_idx = None

                stats = {k: v for k, v in stats.items() if v is not None}
                if ngpu > 1 or distributed:
                    # Apply weighted averaging for loss and stats
                    loss = (loss * weight.type(loss.dtype)).sum()

                    # if distributed, this method can also apply all_reduce()
                    stats, weight = recursive_average(stats, weight, distributed)

                    # Now weight is summation over all workers
                    loss /= weight
                if distributed:
                    # NOTE(kamo): Multiply world_size because DistributedDataParallel
                    # automatically normalizes the gradient by world_size.
                    loss *= torch.distributed.get_world_size()

                loss /= accum_grad

            reporter.register(stats, weight)

            with reporter.measure_time("backward_time"):
                if scaler is not None:
                    # Scales loss.  Calls backward() on scaled loss
                    # to create scaled gradients.
                    # Backward passes under autocast are not recommended.
                    # Backward ops run in the same dtype autocast chose
                    # for corresponding forward ops.
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if iiter % accum_grad == 0:
                if scaler is not None:
                    # Unscales the gradients of optimizer's assigned params in-place
                    for iopt, optimizer in enumerate(optimizers):
                        if optim_idx is not None and iopt != optim_idx:
                            continue
                        scaler.unscale_(optimizer)

                # gradient noise injection
                if grad_noise:
                    add_gradient_noise(
                        model,
                        reporter.get_total_count(),
                        duration=100,
                        eta=1.0,
                        scale_factor=0.55,
                    )

                # compute the gradient norm to check if it is normal or not
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                    norm_type=grad_clip_type,
                )
                # PyTorch<=1.4, clip_grad_norm_ returns float value
                if not isinstance(grad_norm, torch.Tensor):
                    grad_norm = torch.tensor(grad_norm)

                if not torch.isfinite(grad_norm):
                    logging.warning(
                        f"The grad norm is {grad_norm}. Skipping updating the model."
                    )

                    # Must invoke scaler.update() if unscale_() is used in the iteration
                    # to avoid the following error:
                    #   RuntimeError: unscale_() has already been called
                    #   on this optimizer since the last update().
                    # Note that if the gradient has inf/nan values,
                    # scaler.step skips optimizer.step().
                    if scaler is not None:
                        for iopt, optimizer in enumerate(optimizers):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            scaler.step(optimizer)
                            scaler.update()

                else:
                    all_steps_are_invalid = False
                    with reporter.measure_time("optim_step_time"):
                        for iopt, (optimizer, scheduler) in enumerate(
                                zip(optimizers, schedulers)
                        ):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            if scaler is not None:
                                # scaler.step() first unscales the gradients of
                                # the optimizer's assigned params.
                                scaler.step(optimizer)
                                # Updates the scale for next iteration.
                                scaler.update()
                            else:
                                optimizer.step()
                            if isinstance(scheduler, AbsBatchStepScheduler):
                                scheduler.step()
                for iopt, optimizer in enumerate(optimizers):
                    if optim_idx is not None and iopt != optim_idx:
                        continue
                    optimizer.zero_grad()

                # Register lr and train/load time[sec/step],
                # where step refers to accum_grad * mini-batch
                reporter.register(
                    dict(
                        {
                            f"optim{i}_lr{j}": pg["lr"]
                            for i, optimizer in enumerate(optimizers)
                            for j, pg in enumerate(optimizer.param_groups)
                            if "lr" in pg
                        },
                        train_time=time.perf_counter() - start_time,
                    ),
                )
                start_time = time.perf_counter()

                # update num_updates
                if distributed:
                    if hasattr(model.module, "num_updates"):
                        model.module.set_num_updates(model.module.get_num_updates() + 1)
                        options.num_updates = model.module.get_num_updates()
                        if model.module.get_num_updates() >= options.max_update:
                            max_update_stop = True
                else:
                    if hasattr(model, "num_updates"):
                        model.set_num_updates(model.get_num_updates() + 1)
                        options.num_updates = model.get_num_updates()
                        if model.get_num_updates() >= options.max_update:
                            max_update_stop = True

            # NOTE(kamo): Call log_message() after next()
            reporter.next()
            if iiter % log_interval == 0:
                num_updates = options.num_updates if hasattr(options, "num_updates") else None
                logging.info(reporter.log_message(-log_interval, num_updates=num_updates))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                if use_wandb:
                    reporter.wandb_log()

            if max_update_stop:
                break

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        return all_steps_are_invalid, max_update_stop

    @torch.no_grad()
    def validate_one_epoch(
            self,
            model: torch.nn.Module,
            iterator: Iterable[Dict[str, torch.Tensor]],
            reporter: SubReporter,
            options: TrainerOptions,
            distributed_option: DistributedOption,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed

        model.eval()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for (_, batch) in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            retval = model(**batch)
            if isinstance(retval, dict):
                stats = retval["stats"]
                weight = retval["weight"]
            else:
                _, stats, weight = retval
            if ngpu > 1 or distributed:
                # Apply weighted averaging for stats.
                # if distributed, this method can also apply all_reduce()
                stats, weight = recursive_average(stats, weight, distributed)

            reporter.register(stats, weight)
            reporter.next()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)


def build_trainer(
        args,
        model: FunASRModel,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        train_dataloader: AbsIterFactory,
        valid_dataloader: AbsIterFactory,
        distributed_option: DistributedOption
):
    trainer = Trainer(
        args=args,
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        distributed_option=distributed_option
    )
    return trainer
