import math
import os
import time
import torch
import logging
from tqdm import tqdm
from datetime import datetime
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext, contextmanager
from pathlib import Path

from funasr.train_utils.device_funcs import to_device
from funasr.train_utils.recursive_op import recursive_average
from funasr.train_utils.average_nbest_models import average_checkpoints
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

try:
    import wandb
except:
    wandb = None


@contextmanager
def maybe_autocast(enabled):
    if enabled:
        with autocast():
            yield
    else:
        yield


class Trainer:
    """
    A simple trainer class for training a PyTorch model, saving checkpoints at the end of each epoch,
    and optionally resuming from a saved checkpoint.

    Attributes:
        max_epoch (int): Maximum number of epochs for training.
        model (torch.nn.Module): The model to be trained.
        optim (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        dataloader_train (torch.utils.data.DataLoader): DataLoader for the training dataset.
        dataloader_val (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        output_dir (str): Directory where model checkpoints will be saved.
        resume (str, optional): Path to a checkpoint to resume training from.
    """

    def __init__(
        self,
        local_rank,
        use_ddp: bool = False,
        use_fsdp: bool = False,
        use_fp16: bool = False,
        output_dir: str = "./",
        **kwargs,
    ):
        """
        Initializes the Trainer class with the model, optimizer, scheduler, dataloaders, and other settings.

        Args:
            model (torch.nn.Module): The model to be trained.
            optim (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            dataloader_train (torch.utils.data.DataLoader): The DataLoader for the training dataset.
            dataloader_val (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
            **kwargs: Additional keyword arguments:
                      max_epoch (int): The maximum number of epochs for training.
                      output_dir (str): The directory where model checkpoints will be saved. Default is './'.
                      resume (str, optional): The file path to a checkpoint to resume training from.
        """

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.resume = kwargs.get("resume", True)
        self.start_epoch = 0
        self.max_epoch = kwargs.get("max_epoch", 100)
        self.local_rank = local_rank
        self.use_ddp = use_ddp
        self.use_fsdp = use_fsdp
        self.device = kwargs.get("device", "cuda")
        # self.kwargs = kwargs
        self.log_interval = kwargs.get("log_interval", 50)
        self.batch_total = 0
        self.use_fp16 = use_fp16
        self.save_checkpoint_interval = kwargs.get("save_checkpoint_interval", 5000)
        self.validate_interval = kwargs.get("validate_interval", 5000)
        self.keep_nbest_models = kwargs.get("keep_nbest_models", 500)
        self.avg_keep_nbest_models_type = kwargs.get("avg_keep_nbest_models_type", "acc")
        self.avg_nbest_model = kwargs.get("avg_nbest_model", 10)
        self.accum_grad = kwargs.get("accum_grad", 1)
        self.grad_clip = kwargs.get("grad_clip", 10.0)
        self.grad_clip_type = kwargs.get("grad_clip_type", 2.0)

        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1
            logging.warning("distributed is not initialized, only single shard")
        self.rank = rank
        self.world_size = world_size
        self.train_acc_avg = 0.0
        self.train_loss_avg = 0.0
        self.val_acc_avg = 0.0
        self.val_loss_avg = 0.0
        self.best_acc_idx = 0
        self.saved_ckpts = {}
        self.step_or_epoch = -1
        self.best_step_or_epoch = ""
        self.val_acc_step_or_eoch = {}
        self.val_loss_step_or_eoch = {}

        self.reset_gpu_cache = kwargs.get("reset_gpu_cache", False)
        self.start_data_split_i = 0
        self.start_step = 0
        self.step_in_epoch = 0
        self.use_wandb = kwargs.get("use_wandb", False)
        if self.use_wandb:
            wandb.login(key=kwargs.get("wandb_token"))
            wandb.init(
                config=kwargs,
                project=kwargs.get("wandb_project", "my_project"),
                entity=kwargs.get("wandb_team", "my_team"),
                name=kwargs.get("wandb_exp_name", "my_exp"),
                dir=output_dir,
                job_type="training",
                reinit=True,
            )

    def save_checkpoint(
        self,
        epoch,
        step=None,
        model=None,
        optim=None,
        scheduler=None,
        scaler=None,
        step_in_epoch=None,
        **kwargs,
    ):
        """
        Saves a checkpoint containing the model's state, the optimizer's state,
        and the scheduler's state at the end of the given epoch. This method is
        intended to be called at the end of each epoch to save the training progress.

        Args:
            epoch (int): The epoch number at which the checkpoint is being saved.
        """

        step_in_epoch = None if step is None else step_in_epoch
        if self.rank == 0:
            logging.info(f"Save checkpoint: {epoch}, rank: {self.local_rank}\n")
            # self.step_or_epoch += 1
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "saved_ckpts": self.saved_ckpts,
                "val_acc_step_or_eoch": self.val_acc_step_or_eoch,
                "val_loss_step_or_eoch": self.val_loss_step_or_eoch,
                "best_step_or_epoch": self.best_step_or_epoch,
                "avg_keep_nbest_models_type": self.avg_keep_nbest_models_type,
                "step": step,
                "step_in_epoch": step_in_epoch,
                "data_split_i": kwargs.get("data_split_i", 0),
                "data_split_num": kwargs.get("data_split_num", 1),
                "batch_total": self.batch_total,
                "train_loss_avg": kwargs.get("train_loss_avg", 0),
                "train_acc_avg": kwargs.get("train_acc_avg", 0),
            }
            step = step_in_epoch
            if hasattr(model, "module"):
                state["state_dict"] = model.module.state_dict()

            if scaler:
                state["scaler_state"] = scaler.state_dict()
            # Create output directory if it does not exist
            os.makedirs(self.output_dir, exist_ok=True)
            if step is None:
                ckpt_name = f"model.pt.ep{epoch}"
            else:
                ckpt_name = f"model.pt.ep{epoch}.{step}"
            filename = os.path.join(self.output_dir, ckpt_name)
            torch.save(state, filename)

            logging.info(f"\nCheckpoint saved to {filename}\n")
            latest = Path(os.path.join(self.output_dir, f"model.pt"))
            torch.save(state, latest)
            if self.best_step_or_epoch == "":
                self.best_step_or_epoch = ckpt_name

            if self.avg_keep_nbest_models_type == "acc":
                if (
                    self.val_acc_step_or_eoch[ckpt_name]
                    >= self.val_acc_step_or_eoch[self.best_step_or_epoch]
                ):
                    self.best_step_or_epoch = ckpt_name
                    best_ckpt = Path(os.path.join(self.output_dir, f"model.pt.best"))
                    torch.save(state, best_ckpt)
                    logging.info(
                        f"Update best acc: {self.val_acc_step_or_eoch[self.best_step_or_epoch]:.4f}, {best_ckpt}"
                    )
                else:
                    logging.info(
                        f"No improvement in acc: {self.val_acc_step_or_eoch[ckpt_name]:.4f} < {self.val_acc_step_or_eoch[self.best_step_or_epoch]:.4f}, {os.path.join(self.output_dir, self.best_step_or_epoch)}"
                    )
            elif self.avg_keep_nbest_models_type == "loss":
                if (
                    self.val_loss_step_or_eoch[ckpt_name]
                    <= self.val_loss_step_or_eoch[self.best_step_or_epoch]
                ):
                    self.best_step_or_epoch = ckpt_name
                    best_ckpt = Path(os.path.join(self.output_dir, f"model.pt.best"))
                    torch.save(state, best_ckpt)
                    logging.info(
                        f"Update best loss: {self.val_loss_step_or_eoch[self.best_step_or_epoch]:.4f}, {best_ckpt}"
                    )
                else:
                    logging.info(
                        f"No improvement in loss: {self.val_loss_step_or_eoch[ckpt_name]:.4f} > {self.val_loss_step_or_eoch[self.best_step_or_epoch]:.4f}, {os.path.join(self.output_dir, self.best_step_or_epoch)}"
                    )
            else:
                print("Undo")
            self.saved_ckpts[ckpt_name] = getattr(
                self, f"val_{self.avg_keep_nbest_models_type}_step_or_eoch"
            )[ckpt_name]
            if self.keep_nbest_models > 0:
                if len(self.saved_ckpts) > self.keep_nbest_models:
                    if self.avg_keep_nbest_models_type == "acc":
                        key = min(self.saved_ckpts, key=self.saved_ckpts.get)
                    else:
                        key = max(self.saved_ckpts, key=self.saved_ckpts.get)
                    if key in self.saved_ckpts:
                        del self.saved_ckpts[key]
                    filename = os.path.join(self.output_dir, key)
                    logging.info(f"Delete: {filename}")
                    if os.path.exists(filename):
                        os.remove(filename)

        if self.use_ddp or self.use_fsdp:
            dist.barrier()

    def resume_checkpoint(
        self,
        model=None,
        optim=None,
        scheduler=None,
        scaler=None,
    ):
        """
        Resumes training from a checkpoint at the given file path.
        Loads the model's state, the optimizer's state, and the scheduler's state.

        Args:
            resume_path (str): The file path to the checkpoint to resume from.
        """
        if self.resume:
            ckpt = os.path.join(self.output_dir, "model.pt")
            if os.path.isfile(ckpt):
                checkpoint = torch.load(ckpt, map_location="cpu")
                self.start_epoch = checkpoint["epoch"]
                # self.model.load_state_dict(checkpoint['state_dict'])
                src_state = checkpoint["state_dict"]
                dst_state = model.state_dict()
                for k in dst_state.keys():
                    if not k.startswith("module.") and "module." + k in src_state.keys():
                        k_ddp = "module." + k
                    elif k.startswith("module.") and "module." + k not in src_state.keys():
                        k_ddp = k.replace("module.", "", 1)
                    else:
                        k_ddp = k
                    if k_ddp in src_state.keys():
                        dst_state[k] = src_state[k_ddp]
                    else:
                        print(f"Miss key in ckpt: model: {k}, ckpt: {k_ddp}")

                model.load_state_dict(dst_state)
                optim.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                if scaler is not None and "scaler_state" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler_state"])

                self.saved_ckpts = checkpoint["saved_ckpts"]
                self.val_acc_step_or_eoch = (
                    checkpoint["val_acc_step_or_eoch"]
                    if "val_acc_step_or_eoch" in checkpoint
                    else {}
                )
                self.val_loss_step_or_eoch = (
                    checkpoint["val_loss_step_or_eoch"]
                    if "val_loss_step_or_eoch" in checkpoint
                    else {}
                )
                self.best_step_or_epoch = (
                    checkpoint["best_step_or_epoch"] if "best_step_or_epoch" in checkpoint else ""
                )
                self.start_data_split_i = (
                    checkpoint["data_split_i"] if "data_split_i" in checkpoint else 0
                )
                self.batch_total = checkpoint["batch_total"] if "batch_total" in checkpoint else 0
                self.start_step = checkpoint["step"] if "step" in checkpoint else 0
                self.start_step = 0 if self.start_step is None else self.start_step
                self.step_in_epoch = (
                    checkpoint["step_in_epoch"] if "step_in_epoch" in checkpoint else 0
                )
                self.step_in_epoch = 0 if self.step_in_epoch is None else self.step_in_epoch
                print(checkpoint["train_acc_avg"])
                self.train_acc_avg = (
                    checkpoint["train_acc_avg"] if "train_acc_avg" in checkpoint else 0
                )
                self.train_loss_avg = (
                    checkpoint["train_loss_avg"] if "train_loss_avg" in checkpoint else 0
                )
                model.to(self.device)
                print(f"Checkpoint loaded successfully from '{ckpt}'")
            else:
                print(f"No checkpoint found at '{ckpt}', does not resume status!")

        if self.use_ddp or self.use_fsdp:
            dist.barrier()

    def train_epoch(
        self,
        model=None,
        optim=None,
        scheduler=None,
        scaler=None,
        dataloader_train=None,
        dataloader_val=None,
        epoch=None,
        writer=None,
        **kwargs,
    ):
        """
        Defines the training process for a single epoch with gradient accumulation.
        Args:
            epoch (int): The current epoch number.
        """
        if self.use_ddp or self.use_fsdp:
            dist.barrier()
        logging.info(f"Train epoch: {epoch}, rank: {self.rank}\n")
        model.train()

        # Set the number of steps for gradient accumulation
        accum_grad = self.accum_grad
        # Initialize the gradient accumulation
        optim.zero_grad()
        speed_stats = {}

        iterator_stop = torch.tensor(0).to(self.device)

        dataloader_train.batch_sampler.set_epoch(epoch)
        time_beg = time.perf_counter()
        time5 = time_beg
        for batch_idx, batch in enumerate(dataloader_train):
            if self.use_ddp or self.use_fsdp:
                dist.all_reduce(iterator_stop, dist.ReduceOp.SUM)
                if iterator_stop > 0:
                    break
            self.batch_total += 1
            self.step_in_epoch += 1
            time1 = time.perf_counter()
            speed_stats["data_load"] = f"{time1-time_beg:0.3f}"

            batch = to_device(batch, self.device)

            my_context = nullcontext
            if self.use_ddp or self.use_fsdp:
                my_context = model.no_sync if batch_idx % accum_grad != 0 else my_context
            with my_context():
                time2 = time.perf_counter()
                with maybe_autocast(self.use_fp16):
                    retval = model(**batch)

                    if (
                        self.reset_gpu_cache
                        and (torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024) > 70
                    ):
                        torch.cuda.empty_cache()

                loss, stats, weight = retval
                stats = {k: v for k, v in stats.items() if v is not None}
                if self.use_ddp or self.use_fsdp:
                    # Apply weighted averaging for loss and stats
                    loss = (loss * weight.type(loss.dtype)).sum()
                    # if distributed, this method can also apply all_reduce()
                    # stats, weight = recursive_average(stats, weight, distributed=True)
                    if self.use_ddp or self.use_fsdp:
                        dist.all_reduce(weight, op=dist.ReduceOp.SUM)
                    # Now weight is summation over all workers
                    loss /= weight.sum()  # shape:[1] -> shape:[]
                    # Multiply world_size because DistributedDataParallel
                    # automatically normalizes the gradient by world_size.
                    loss *= self.world_size
                # loss *= self.world_size
                # Scale the loss since we're not updating for every mini-batch
                loss = loss / accum_grad

                time3 = time.perf_counter()
                speed_stats["forward_time"] = f"{time3 - time2:0.3f}"
                if self.use_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                time4 = time.perf_counter()
                speed_stats["backward_and_AllReaduce_time"] = f"{time4 - time3:0.3f}"

                self.train_loss_avg = (
                    self.train_loss_avg * (batch_idx + kwargs.get("start_step", 0))
                    + loss.detach().cpu().item()
                ) / (batch_idx + kwargs.get("start_step", 0) + 1)
                if "acc" in stats:
                    self.train_acc_avg = (
                        self.train_acc_avg * (batch_idx + kwargs.get("start_step", 0))
                        + stats["acc"].detach().cpu().item()
                    ) / (batch_idx + kwargs.get("start_step", 0) + 1)

            # Perform an optimizer step only after accumulating enough gradients
            if (batch_idx + 1) % accum_grad == 0:
                # Perform gradient clipping if it is set
                if self.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=self.grad_clip,
                        norm_type=self.grad_clip_type,
                    )
                    if not torch.isfinite(grad_norm):
                        logging.warning(
                            f"The grad norm is {grad_norm}. Skipping updating the model."
                        )
                        optim.zero_grad()  # Reset gradients
                        continue

                # Execute an optimization step (update model parameters)
                if self.use_ddp or self.use_fsdp:
                    dist.barrier()
                if self.use_fp16:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                scheduler.step()
                # Clear gradients for the next accumulation stage
                optim.zero_grad(set_to_none=True)

                if self.use_ddp or self.use_fsdp:
                    train_loss_avg = torch.tensor(self.train_loss_avg, dtype=torch.float32).to(
                        self.device
                    )
                    train_acc_avg = torch.tensor(self.train_acc_avg, dtype=torch.float32).to(
                        self.device
                    )
                    dist.all_reduce(train_loss_avg, op=dist.ReduceOp.SUM)
                    dist.all_reduce(train_acc_avg, op=dist.ReduceOp.SUM)
                    self.train_loss_avg = train_loss_avg.detach().cpu().item() / self.world_size
                    self.train_acc_avg = train_acc_avg.detach().cpu().item() / self.world_size

                total_time = f"{(time.perf_counter() - time5)/accum_grad:0.3f}"
                time5 = time.perf_counter()

                speed_stats["optim_time"] = f"{time5 - time4:0.3f}"

                speed_stats["total_time"] = total_time
                lr = scheduler.get_last_lr()[0]
                batch_num_epoch = 1
                if hasattr(dataloader_train, "__len__"):
                    batch_num_epoch = len(dataloader_train)
                self.log(
                    epoch,
                    batch_idx,
                    log_step=batch_idx + kwargs.get("start_step", 0),
                    step_in_epoch=self.step_in_epoch,
                    batch_num_epoch=batch_num_epoch,
                    lr=lr,
                    loss=loss.detach().cpu().item(),
                    speed_stats=speed_stats,
                    stats=stats,
                    writer=writer,
                    tag="train",
                    data_split_i=kwargs.get("data_split_i", 0),
                    data_split_num=kwargs.get("data_split_num", 1),
                )

            if self.step_in_epoch % self.validate_interval == 0:
                self.validate_epoch(
                    model=model,
                    dataloader_val=dataloader_val,
                    epoch=epoch,
                    writer=writer,
                    step=batch_idx + 1,
                    step_in_epoch=self.step_in_epoch,
                )

            if self.step_in_epoch % self.save_checkpoint_interval == 0:
                self.save_checkpoint(
                    epoch,
                    model=model,
                    optim=optim,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=batch_idx + 1,
                    step_in_epoch=self.step_in_epoch,
                    data_split_i=kwargs.get("data_split_i", 0),
                    data_split_num=kwargs.get("data_split_num", 1),
                    train_loss_avg=self.train_loss_avg,
                    train_acc_avg=self.train_acc_avg,
                )

            time_beg = time.perf_counter()
        else:
            if self.use_ddp or self.use_fsdp:
                iterator_stop.fill_(1)
                dist.all_reduce(iterator_stop, dist.ReduceOp.SUM)

        if self.use_ddp or self.use_fsdp:
            dist.barrier()
            iterator_stop = torch.tensor(0).to(self.device)

    def validate_epoch(
        self,
        model=None,
        dataloader_val=None,
        epoch=None,
        writer=None,
        **kwargs,
    ):
        """
        Defines the validation process for a single epoch.
        Should be implemented with the actual model validation steps.

        Args:
            epoch (int): The current epoch number.
        """
        if self.use_ddp or self.use_fsdp:
            dist.barrier()
        logging.info(f"Validate epoch: {epoch}, rank: {self.rank}\n")
        model.eval()

        with torch.no_grad():

            speed_stats = {}
            time5 = time.perf_counter()
            iterator_stop = torch.tensor(0).to(self.device)
            dataloader_val.batch_sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(dataloader_val):
                if self.use_ddp or self.use_fsdp:
                    dist.all_reduce(iterator_stop, dist.ReduceOp.SUM)
                    if iterator_stop > 0:
                        break
                time1 = time.perf_counter()
                speed_stats["data_load"] = f"{time1 - time5:0.3f}"
                batch = to_device(batch, self.device)
                time2 = time.perf_counter()
                retval = model(**batch)
                time3 = time.perf_counter()
                speed_stats["forward_time"] = f"{time3 - time2:0.3f}"
                loss, stats, weight = retval
                stats = {k: v for k, v in stats.items() if v is not None}
                if self.use_ddp or self.use_fsdp:
                    # Apply weighted averaging for loss and stats
                    loss = (loss * weight.type(loss.dtype)).sum()
                    # if distributed, this method can also apply all_reduce()
                    # stats, weight = recursive_average(stats, weight, distributed=True)
                    if self.use_ddp or self.use_fsdp:
                        dist.all_reduce(weight, op=dist.ReduceOp.SUM)
                    # Now weight is summation over all workers
                    loss /= weight.sum()  # shape:[1] -> shape:[]
                    # Multiply world_size because DistributedDataParallel
                    # automatically normalizes the gradient by world_size.
                    loss *= self.world_size
                # Scale the loss since we're not updating for every mini-batch
                loss = loss
                time4 = time.perf_counter()

                self.val_loss_avg = (self.val_loss_avg * batch_idx + loss.detach().cpu().item()) / (
                    batch_idx + 1
                )
                if "acc" in stats:
                    self.val_acc_avg = (
                        self.val_acc_avg * batch_idx + stats["acc"].detach().cpu().item()
                    ) / (batch_idx + 1)
                if self.use_ddp or self.use_fsdp:
                    val_loss_avg = torch.tensor(self.val_loss_avg, dtype=torch.float32).to(
                        self.device
                    )
                    val_acc_avg = torch.tensor(self.val_acc_avg, dtype=torch.float32).to(
                        self.device
                    )
                    dist.all_reduce(val_loss_avg, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_acc_avg, op=dist.ReduceOp.SUM)
                    self.val_loss_avg = val_loss_avg.detach().cpu().item() / self.world_size
                    self.val_acc_avg = val_acc_avg.detach().cpu().item() / self.world_size
                time5 = time.perf_counter()
                batch_num_epoch = 1
                if hasattr(dataloader_val, "__len__"):
                    batch_num_epoch = len(dataloader_val)
                self.log(
                    epoch,
                    batch_idx,
                    batch_num_epoch=batch_num_epoch,
                    lr=0.0,
                    loss=loss.detach().cpu().item(),
                    speed_stats=speed_stats,
                    stats=stats,
                    writer=writer,
                    tag="val",
                )

            else:
                if self.use_ddp or self.use_fsdp:
                    iterator_stop.fill_(1)
                    dist.all_reduce(iterator_stop, dist.ReduceOp.SUM)

        if kwargs.get("step_in_epoch", None) is None:
            ckpt_name = f"model.pt.ep{epoch}"
        else:
            ckpt_name = f'model.pt.ep{epoch}.{kwargs.get("step_in_epoch")}'
        self.val_acc_step_or_eoch[ckpt_name] = self.val_acc_avg
        self.val_loss_step_or_eoch[ckpt_name] = self.val_loss_avg
        model.train()

        if self.use_ddp or self.use_fsdp:
            dist.barrier()
            iterator_stop = torch.tensor(0).to(self.device)

    def log(
        self,
        epoch=0,
        batch_idx=0,
        step_in_epoch=0,
        batch_num_epoch=-1,
        lr=0.0,
        loss=0.0,
        speed_stats=None,
        stats=None,
        writer=None,
        tag="train",
        data_split_i=0,
        data_split_num=1,
        log_step=None,
        **kwargs,
    ):

        if (batch_idx + 1) % self.log_interval == 0:
            batch_idx = log_step if log_step is not None else batch_idx
            gpu_info = (
                "GPU, memory: usage: {:.3f} GB, "
                "peak: {:.3f} GB, "
                "cache: {:.3f} GB, "
                "cache_peak: {:.3f} GB".format(
                    torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
                    torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
                    torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
                    torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
                )
            )

            loss_avg_epoch = getattr(self, f"{tag}_loss_avg")
            acc_avg_epoch = getattr(self, f"{tag}_acc_avg")
            description = (
                f"{tag}, "
                f"rank: {self.rank}, "
                f"epoch: {epoch}/{self.max_epoch}, "
                f"data_slice: {data_split_i}/{data_split_num}, "
                f"step_in_slice: {batch_idx + 1}/{batch_num_epoch}, step_in_epoch: {step_in_epoch}, total step: {self.batch_total}, "
                f"(loss_avg_rank: {loss:.3f}), "
                f"(loss_avg_slice: {loss_avg_epoch:.3f}), "
                f"(ppl_avg_slice: {math.exp(loss_avg_epoch):.3e}), "
                f"(acc_avg_slice: {acc_avg_epoch:.3f}), "
                f"(lr: {lr:.3e}), "
                f"{[(k, round(v.detach().cpu().item(), 3)) for k, v in stats.items()]}, "
                f"{speed_stats}, "
                f"{gpu_info}"
            )
            logging.info(description)

            description_dict = {
                f"rank{self.rank}_loss/{tag}": loss,
                f"rank{self.rank}_lr/{tag}": lr,
            }

            if writer is not None:
                writer.add_scalar(f"rank{self.rank}_loss/{tag}", loss, self.batch_total)
                writer.add_scalar(f"rank{self.rank}_lr/{tag}", lr, self.batch_total)
                for key, var in stats.items():
                    writer.add_scalar(
                        f"stats_rank{self.rank}_{key}/{tag}", var.item(), self.batch_total
                    )
                    description_dict[f"stats_rank{self.rank}_{key}/{tag}"] = var.item()
                for key, var in speed_stats.items():
                    writer.add_scalar(
                        f"stats_rank{self.rank}_{key}/{tag}", eval(var), self.batch_total
                    )
                    description_dict[f"stats_rank{self.rank}_{key}/{tag}"] = eval(var)
            if self.use_wandb and wandb is not None:
                wandb.log(
                    description_dict,
                    setp=self.batch_total,
                )

    def close(self, writer=None):

        if self.use_ddp or self.use_fsdp:
            dist.barrier()

        if writer is not None:
            writer.close()

        if self.use_ddp or self.use_fsdp:
            torch.distributed.destroy_process_group()
