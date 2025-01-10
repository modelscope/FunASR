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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from funasr.train_utils.device_funcs import to_device
from funasr.train_utils.recursive_op import recursive_average
from funasr.train_utils.average_nbest_models import average_checkpoints
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import funasr.utils.misc as misc_utils

try:
    import wandb
except:
    wandb = None


@contextmanager
def maybe_autocast(dtype=None, use_deepspeed=False):
    if use_deepspeed:
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False):
            yield
    else:
        if dtype == torch.float16 or dtype == torch.bfloat16:
            yield
            # with autocast(enabled=True, dtype=dtype):
            #     yield
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
        rank=0,
        local_rank=0,
        world_size=1,
        use_ddp: bool = False,
        use_fsdp: bool = False,
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_deepspeed: bool = False,
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
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        self.use_fsdp = use_fsdp

        self.device = kwargs.get("device", "cuda")

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.resume = kwargs.get("resume", True)
        self.start_epoch = 0
        self.max_epoch = kwargs.get("max_epoch", 100)

        # self.kwargs = kwargs
        self.log_interval = kwargs.get("log_interval", 50)
        self.batch_total = 0
        self.dtype = torch.float32
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        if self.use_fp16:
            self.dtype = torch.float16
        if self.use_bf16:
            self.dtype = torch.bfloat16
        self.save_checkpoint_interval = kwargs.get("save_checkpoint_interval", 5000)
        self.validate_interval = kwargs.get("validate_interval", 5000)
        self.keep_nbest_models = kwargs.get("keep_nbest_models", 500)
        self.avg_keep_nbest_models_type = kwargs.get("avg_keep_nbest_models_type", "acc")
        self.avg_nbest_model = kwargs.get("avg_nbest_model", 10)
        self.accum_grad = kwargs.get("accum_grad", 1)
        self.grad_clip = kwargs.get("grad_clip", 10.0)
        self.grad_clip_type = kwargs.get("grad_clip_type", 2.0)

        self.train_acc_avg = 0.0
        self.train_loss_avg = 0.0
        self.val_acc_avg = 0.0
        self.val_loss_avg = 0.0
        self.best_acc_idx = 0
        self.saved_ckpts = {}
        self.step_or_epoch = -1
        self.best_step_or_epoch = ""
        self.val_acc_step_or_epoch = {}
        self.val_loss_step_or_epoch = {}

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
        tensorboard_dir = os.path.join(output_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        try:
            from tensorboardX import SummaryWriter

            self.writer = SummaryWriter(tensorboard_dir)  # if trainer.rank == 0 else None
        except:
            self.writer = None

        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = kwargs.get("deepspeed_config", "")
        excludes = kwargs.get("excludes", None)
        if excludes is not None:
            if isinstance(excludes, str):
                excludes = excludes.split(",")
        self.excludes = excludes
        effective_save_name_excludes = kwargs.get("effective_save_name_excludes", None)
        if effective_save_name_excludes is not None:
            if isinstance(effective_save_name_excludes, str):
                effective_save_name_excludes = effective_save_name_excludes.split(",")
        self.effective_save_name_excludes = effective_save_name_excludes

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
        if self.use_ddp or self.use_fsdp:
            dist.barrier()
        step_in_epoch = None if step is None else step_in_epoch
        if self.use_deepspeed:

            logging.info(f"Save checkpoint: {epoch}, rank: {self.local_rank}\n")
            # self.step_or_epoch += 1
            state = {
                "epoch": epoch,
                # "state_dict": model.state_dict(),
                # "optimizer": optim.state_dict(),
                # "scheduler": scheduler.state_dict(),
                "saved_ckpts": self.saved_ckpts,
                "val_acc_step_or_epoch": self.val_acc_step_or_epoch,
                "val_loss_step_or_epoch": self.val_loss_step_or_epoch,
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

            # torch.save(state, filename)
            with torch.no_grad():
                model.save_checkpoint(save_dir=self.output_dir, tag=ckpt_name, client_state=state)
            logging.info(f"\nCheckpoint saved to {filename}\n")
            latest = Path(os.path.join(self.output_dir, f"model.pt"))
            # torch.save(state, latest)
            with torch.no_grad():
                model.save_checkpoint(save_dir=self.output_dir, tag=f"model.pt", client_state=state)
            if self.best_step_or_epoch == "":
                self.best_step_or_epoch = ckpt_name

            if self.avg_keep_nbest_models_type == "acc":
                if (
                    self.val_acc_step_or_epoch[ckpt_name]
                    >= self.val_acc_step_or_epoch[self.best_step_or_epoch]
                ):
                    self.best_step_or_epoch = ckpt_name
                    best_ckpt = Path(os.path.join(self.output_dir, f"model.pt.best"))
                    # torch.save(state, best_ckpt)
                    with torch.no_grad():
                        model.save_checkpoint(
                            save_dir=self.output_dir, tag=f"model.pt.best", client_state=state
                        )
                    logging.info(
                        f"Update best acc: {self.val_acc_step_or_epoch[self.best_step_or_epoch]:.4f}, {best_ckpt}"
                    )
                else:
                    logging.info(
                        f"No improvement in acc: {self.val_acc_step_or_epoch[ckpt_name]:.4f} < {self.val_acc_step_or_epoch[self.best_step_or_epoch]:.4f}, {os.path.join(self.output_dir, self.best_step_or_epoch)}"
                    )
            elif self.avg_keep_nbest_models_type == "loss":
                if (
                    self.val_loss_step_or_epoch[ckpt_name]
                    <= self.val_loss_step_or_epoch[self.best_step_or_epoch]
                ):
                    self.best_step_or_epoch = ckpt_name
                    best_ckpt = Path(os.path.join(self.output_dir, f"model.pt.best"))
                    # torch.save(state, best_ckpt)
                    with torch.no_grad():
                        model.save_checkpoint(
                            save_dir=self.output_dir, tag=f"model.pt.best", client_state=state
                        )
                    logging.info(
                        f"Update best loss: {self.val_loss_step_or_epoch[self.best_step_or_epoch]:.4f}, {best_ckpt}"
                    )
                else:
                    logging.info(
                        f"No improvement in loss: {self.val_loss_step_or_epoch[ckpt_name]:.4f} > {self.val_loss_step_or_epoch[self.best_step_or_epoch]:.4f}, {os.path.join(self.output_dir, self.best_step_or_epoch)}"
                    )
            else:
                print("Undo")
            self.saved_ckpts[ckpt_name] = getattr(
                self, f"val_{self.avg_keep_nbest_models_type}_step_or_epoch"
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
                        # os.remove(filename)
                        misc_utils.smart_remove(filename)

        elif self.use_fsdp:
            pass
        elif self.rank == 0:
            logging.info(
                f"Save checkpoint: {epoch}, rank: {self.rank}, local_rank: {self.local_rank}\n"
            )
            # self.step_or_epoch += 1
            state = {
                "epoch": epoch,
                "optimizer": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "saved_ckpts": self.saved_ckpts,
                "val_acc_step_or_epoch": self.val_acc_step_or_epoch,
                "val_loss_step_or_epoch": self.val_loss_step_or_epoch,
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
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            if self.effective_save_name_excludes is not None:
                logging.info(f"effective_save_name_excludes: {self.effective_save_name_excludes}")
                dst_state_dict = {}
                for k in state_dict.keys():
                    for k_ex in self.effective_save_name_excludes:
                        k_tmp = k.replace("module.", "")
                        if k.startswith(k_ex):
                            logging.info(f"key: {k} matching: {k_ex}, not save it")
                            break
                    else:
                        dst_state_dict[k] = state_dict[k]
                state["state_dict"] = dst_state_dict
            else:
                state["state_dict"] = state_dict

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
                    self.val_acc_step_or_epoch[ckpt_name]
                    >= self.val_acc_step_or_epoch[self.best_step_or_epoch]
                ):
                    self.best_step_or_epoch = ckpt_name
                    best_ckpt = Path(os.path.join(self.output_dir, f"model.pt.best"))
                    torch.save(state, best_ckpt)
                    logging.info(
                        f"Update best acc: {self.val_acc_step_or_epoch[self.best_step_or_epoch]:.4f}, {best_ckpt}"
                    )
                else:
                    logging.info(
                        f"No improvement in acc: {self.val_acc_step_or_epoch[ckpt_name]:.4f} < {self.val_acc_step_or_epoch[self.best_step_or_epoch]:.4f}, {os.path.join(self.output_dir, self.best_step_or_epoch)}"
                    )
            elif self.avg_keep_nbest_models_type == "loss":
                if (
                    self.val_loss_step_or_epoch[ckpt_name]
                    <= self.val_loss_step_or_epoch[self.best_step_or_epoch]
                ):
                    self.best_step_or_epoch = ckpt_name
                    best_ckpt = Path(os.path.join(self.output_dir, f"model.pt.best"))
                    torch.save(state, best_ckpt)
                    logging.info(
                        f"Update best loss: {self.val_loss_step_or_epoch[self.best_step_or_epoch]:.4f}, {best_ckpt}"
                    )
                else:
                    logging.info(
                        f"No improvement in loss: {self.val_loss_step_or_epoch[ckpt_name]:.4f} > {self.val_loss_step_or_epoch[self.best_step_or_epoch]:.4f}, {os.path.join(self.output_dir, self.best_step_or_epoch)}"
                    )
            else:
                print("Undo")
            self.saved_ckpts[ckpt_name] = getattr(
                self, f"val_{self.avg_keep_nbest_models_type}_step_or_epoch"
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
                        # os.remove(filename)
                        misc_utils.smart_remove(filename)

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

            if self.use_deepspeed:
                ckpt = os.path.join(self.output_dir, "model.pt")
                if os.path.exists(ckpt):
                    _, checkpoint = model.load_checkpoint(self.output_dir, "model.pt")
                    self.start_epoch = checkpoint["epoch"]
                    self.saved_ckpts = checkpoint["saved_ckpts"]
                    self.val_acc_step_or_epoch = (
                        checkpoint["val_acc_step_or_epoch"]
                        if "val_acc_step_or_epoch" in checkpoint
                        else {}
                    )
                    self.val_loss_step_or_epoch = (
                        checkpoint["val_loss_step_or_epoch"]
                        if "val_loss_step_or_epoch" in checkpoint
                        else {}
                    )
                    self.best_step_or_epoch = (
                        checkpoint["best_step_or_epoch"]
                        if "best_step_or_epoch" in checkpoint
                        else ""
                    )
                    self.start_data_split_i = (
                        checkpoint["data_split_i"] if "data_split_i" in checkpoint else 0
                    )
                    self.batch_total = (
                        checkpoint["batch_total"] if "batch_total" in checkpoint else 0
                    )
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
            else:

                ckpt = os.path.join(self.output_dir, "model.pt")
                if os.path.isfile(ckpt):
                    checkpoint = torch.load(ckpt, map_location="cpu")
                    self.start_epoch = checkpoint["epoch"]
                    # self.model.load_state_dict(checkpoint['state_dict'])
                    src_state = checkpoint["state_dict"]
                    dst_state = model.state_dict()
                    for k in dst_state.keys():
                        excludes_flag = False
                        if self.excludes is not None:
                            for k_ex in self.excludes:
                                k_tmp = k.replace("module.", "")
                                if k_tmp.startswith(k_ex):
                                    logging.info(f"key: {k} matching: {k_ex}, excluded")
                                    excludes_flag = True
                                    break
                        if excludes_flag:
                            continue
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
                    self.val_acc_step_or_epoch = (
                        checkpoint["val_acc_step_or_epoch"]
                        if "val_acc_step_or_epoch" in checkpoint
                        else {}
                    )
                    self.val_loss_step_or_epoch = (
                        checkpoint["val_loss_step_or_epoch"]
                        if "val_loss_step_or_epoch" in checkpoint
                        else {}
                    )
                    self.best_step_or_epoch = (
                        checkpoint["best_step_or_epoch"]
                        if "best_step_or_epoch" in checkpoint
                        else ""
                    )
                    self.start_data_split_i = (
                        checkpoint["data_split_i"] if "data_split_i" in checkpoint else 0
                    )
                    self.batch_total = (
                        checkpoint["batch_total"] if "batch_total" in checkpoint else 0
                    )
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
        **kwargs,
    ):
        """
        Defines the training process for a single epoch with gradient accumulation.
        Args:
            epoch (int): The current epoch number.
        """
        if self.use_ddp or self.use_fsdp or self.use_deepspeed:
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
            self.batch_total += 1
            self.step_in_epoch += 1
            loss_dict = {
                "speed_stats": {},
                "epoch": epoch,
                "batch_idx": batch_idx,
                "data_split_i": kwargs.get("data_split_i", 0),
                "data_split_num": kwargs.get("data_split_num", 1),
                "log_step": batch_idx + kwargs.get("start_step", 0),
                "batch_total": self.batch_total,
                "step_in_epoch": self.step_in_epoch,
            }

            time1 = time.perf_counter()
            loss_dict["speed_stats"]["data_load"] = f"{time1-time_beg:0.3f}"

            batch = to_device(batch, self.device)

            my_context = nullcontext
            if self.use_ddp or self.use_fsdp:
                my_context = model.no_sync if batch_idx % accum_grad != 0 else my_context
            with my_context():
                time2 = time.perf_counter()

                self.forward_step(model, batch, loss_dict=loss_dict)

                time3 = time.perf_counter()
                loss_dict["speed_stats"]["forward_time"] = f"{time3 - time2:0.3f}"
                self.backward_step(model, scaler, loss_dict=loss_dict)

                time4 = time.perf_counter()
                loss_dict["speed_stats"]["backward_time"] = f"{time4 - time3:0.3f}"

            self.update_step(model, optim, scheduler, scaler, loss_dict=loss_dict)
            total_time = f"{(time.perf_counter() - time5):0.3f}"
            time5 = time.perf_counter()

            loss_dict["speed_stats"]["optim_time"] = f"{time5 - time4:0.3f}"

            loss_dict["speed_stats"]["total_time"] = total_time

            loss_dict["lr"] = scheduler.get_last_lr()[0]
            loss_dict["batch_num_epoch"] = len(dataloader_train)

            self.train_loss_avg = (
                self.train_loss_avg * batch_idx + loss_dict["loss"].detach().cpu().item()
            ) / (batch_idx + 1)
            if "acc" in loss_dict["stats"]:
                self.train_acc_avg = (
                    self.train_acc_avg * batch_idx + loss_dict["stats"]["acc"].detach().cpu().item()
                ) / (batch_idx + 1)

            self.log(loss_dict, tag="train")

            if self.step_in_epoch % self.validate_interval == 0:
                self.validate_epoch(
                    model=model,
                    dataloader_val=dataloader_val,
                    epoch=epoch,
                    writer=self.writer,
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

        if self.use_ddp or self.use_fsdp or self.use_deepspeed:
            train_loss_avg = torch.tensor(self.train_loss_avg, dtype=torch.float32).to(self.device)
            train_acc_avg = torch.tensor(self.train_acc_avg, dtype=torch.float32).to(self.device)
            dist.all_reduce(train_loss_avg, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_acc_avg, op=dist.ReduceOp.SUM)
            self.train_loss_avg = train_loss_avg.detach().cpu().item() / self.world_size
            self.train_acc_avg = train_acc_avg.detach().cpu().item() / self.world_size

    def forward_step(self, model, batch, loss_dict={}):
        with maybe_autocast(dtype=self.dtype, use_deepspeed=self.use_deepspeed):
            retval = model(**batch)

        loss, stats, weight = retval
        stats = {k: v for k, v in stats.items() if v is not None}

        loss_dict["loss"] = loss
        loss_dict["stats"] = stats
        loss_dict["weight"] = weight

    def backward_step(self, model, scaler, loss_dict={}):
        loss = loss_dict["loss"]

        if self.use_deepspeed:
            scaled_loss = model.backward(loss)
        else:
            loss = loss / self.accum_grad
            if self.use_fp16 or self.use_bf16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

    def update_step(self, model, optim, scheduler, scaler, loss_dict=None):
        batch_idx = loss_dict["batch_idx"]
        if self.use_deepspeed:
            model.step()
        else:
            if (batch_idx + 1) % self.accum_grad == 0:
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
                        return

                # Execute an optimization step (update model parameters)
                if self.use_ddp or self.use_fsdp:
                    dist.barrier()
                if self.use_fp16 or self.use_bf16:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                scheduler.step()
                # Clear gradients for the next accumulation stage
                optim.zero_grad(set_to_none=True)

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
        if self.use_ddp or self.use_fsdp or self.use_deepspeed:
            dist.barrier()
        logging.info(f"Validate epoch: {epoch}, rank: {self.rank}\n")
        model.eval()

        with torch.no_grad():

            speed_stats = {}
            time_beg = time.perf_counter()
            time5 = time_beg

            dataloader_val.batch_sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(dataloader_val):

                loss_dict = {
                    "speed_stats": {},
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "data_split_i": kwargs.get("data_split_i", 0),
                    "data_split_num": kwargs.get("data_split_num", 1),
                    "log_step": batch_idx + kwargs.get("start_step", 0),
                    "batch_total": batch_idx + 1,
                    "step_in_epoch": batch_idx + 1,
                    "lr": 0.0,
                }

                time1 = time.perf_counter()
                loss_dict["speed_stats"]["data_load"] = f"{time1 - time_beg:0.3f}"

                batch = to_device(batch, self.device)

                time2 = time.perf_counter()

                self.forward_step(model, batch, loss_dict=loss_dict)

                time3 = time.perf_counter()
                loss_dict["speed_stats"]["forward_time"] = f"{time3 - time2:0.3f}"

                total_time = f"{(time.perf_counter() - time5):0.3f}"
                time5 = time.perf_counter()

                loss_dict["speed_stats"]["total_time"] = total_time

                loss_dict["batch_num_epoch"] = len(dataloader_val)

                self.log(loss_dict, tag="val")
                time_beg = time.perf_counter()
                self.val_loss_avg = (
                    self.val_loss_avg * batch_idx + loss_dict["loss"].detach().cpu().item()
                ) / (batch_idx + 1)
                if "acc" in loss_dict["stats"]:
                    self.val_acc_avg = (
                        self.val_acc_avg * batch_idx
                        + loss_dict["stats"]["acc"].detach().cpu().item()
                    ) / (batch_idx + 1)

            if self.use_ddp or self.use_fsdp or self.use_deepspeed:
                val_loss_avg = torch.tensor(self.val_loss_avg, dtype=torch.float32).to(self.device)
                val_acc_avg = torch.tensor(self.val_acc_avg, dtype=torch.float32).to(self.device)
                dist.all_reduce(val_loss_avg, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_acc_avg, op=dist.ReduceOp.SUM)
                self.val_loss_avg = val_loss_avg.detach().cpu().item() / self.world_size
                self.val_acc_avg = val_acc_avg.detach().cpu().item() / self.world_size

        if kwargs.get("step_in_epoch", None) is None:
            ckpt_name = f"model.pt.ep{epoch}"
        else:
            ckpt_name = f'model.pt.ep{epoch}.{kwargs.get("step_in_epoch")}'
        self.val_acc_step_or_epoch[ckpt_name] = self.val_acc_avg
        self.val_loss_step_or_epoch[ckpt_name] = self.val_loss_avg

        if self.use_ddp or self.use_fsdp or self.use_deepspeed:
            dist.barrier()

        model.train()

    def log(
        self,
        loss_dict: dict = None,
        tag="train",
        **kwargs,
    ):
        loss = loss_dict["loss"].detach().cpu().item()
        epoch = loss_dict["epoch"]
        batch_idx = loss_dict["batch_idx"]
        step_in_epoch = loss_dict["step_in_epoch"]
        batch_total = loss_dict["batch_total"]
        batch_num_epoch = loss_dict["batch_num_epoch"]
        lr = loss_dict["lr"]

        speed_stats = loss_dict["speed_stats"]
        stats = loss_dict["stats"]
        data_split_i = loss_dict["data_split_i"]
        data_split_num = loss_dict["data_split_num"]
        log_step = loss_dict.get("log_step", None)

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
                f"step_in_slice: {batch_idx + 1}/{batch_num_epoch}, step_in_epoch: {step_in_epoch}, total step: {batch_total}, "
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

            writer = self.writer
            if writer is not None:
                writer.add_scalar(f"rank{self.rank}_loss/{tag}", loss, batch_total)
                writer.add_scalar(f"rank{self.rank}_lr/{tag}", lr, batch_total)
                for key, var in stats.items():
                    writer.add_scalar(f"stats_rank{self.rank}_{key}/{tag}", var.item(), batch_total)
                    description_dict[f"stats_rank{self.rank}_{key}/{tag}"] = var.item()
                for key, var in speed_stats.items():
                    writer.add_scalar(f"stats_rank{self.rank}_{key}/{tag}", eval(var), batch_total)
                    description_dict[f"stats_rank{self.rank}_{key}/{tag}"] = eval(var)
            if self.use_wandb and wandb is not None:
                wandb.log(
                    description_dict,
                    setp=batch_total,
                )

    def close(self, writer=None):

        if self.use_ddp or self.use_fsdp:
            dist.barrier()

        if writer is not None:
            writer.close()

        if self.use_ddp or self.use_fsdp:
            torch.distributed.destroy_process_group()

    def warp_model(self, model, **kwargs):

        if self.use_deepspeed:
            from deepspeed.runtime.zero.stage_1_and_2 import (
                estimate_zero2_model_states_mem_needs_all_live,
            )
            from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
            from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

            # NOTE(xcsong): look in detail how the memory estimator API works:
            #   https://deepspeed.readthedocs.io/en/latest/memory.html#discussion
            if int(os.environ.get("RANK", 0)) == 0:
                logging.info("Estimating model states memory needs (zero2)...")
                estimate_zero2_model_states_mem_needs_all_live(
                    model,
                    num_gpus_per_node=local_world_size,
                    num_nodes=world_size // local_world_size,
                )
                logging.info("Estimating model states memory needs (zero3)...")
                estimate_zero3_model_states_mem_needs_all_live(
                    model,
                    num_gpus_per_node=local_world_size,
                    num_nodes=world_size // local_world_size,
                )
            device = None  # Init device later
            pass  # Init DeepSpeed later

        elif self.use_ddp:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            model = model.cuda(local_rank)
            model = DDP(
                model,
                device_ids=[local_rank],
                find_unused_parameters=kwargs.get("train_conf", {}).get(
                    "find_unused_parameters", False
                ),
            )

        else:
            model = model.to(device=kwargs.get("device", "cuda"))

        return model

    def warp_optim_scheduler(self, model, **kwargs):
        from funasr.optimizers import optim_classes
        from funasr.schedulers import scheduler_classes
        from omegaconf import OmegaConf, DictConfig
        import json

        # optim
        logging.info("Build optim")
        optim = kwargs.get("optim", "adam")
        assert optim in optim_classes
        optim_class = optim_classes.get(optim)
        optim = optim_class(model.parameters(), **kwargs.get("optim_conf"))

        # scheduler
        logging.info("Build scheduler")
        scheduler = kwargs.get("scheduler", "warmuplr")
        assert scheduler in scheduler_classes
        scheduler_class = scheduler_classes.get(scheduler)
        scheduler = scheduler_class(optim, **kwargs.get("scheduler_conf"))

        if self.use_deepspeed:
            import deepspeed

            args = OmegaConf.create({"deepspeed_config": self.deepspeed_config})
            with open(self.deepspeed_config, "r") as fin:
                ds_configs = json.load(fin)

            if "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
                self.dtype = torch.bfloat16

            if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
                self.dtype = torch.float16
            if "optimizer" in ds_configs:
                # NOTE(xcsong): Disable custom optimizer if it is set in ds_config,
                # extremely useful when enable cpu_offload, DeepspeedCpuAdam
                # could be 4~5x faster than torch native adam
                optim = None
                if "scheduler" in ds_configs:
                    scheduler = None
                else:

                    def scheduler(opt):
                        return scheduler_class(opt, **kwargs.get("scheduler_conf"))

            model, optimizer, _, scheduler = deepspeed.initialize(
                args=args,
                model=model,
                optimizer=optim,
                lr_scheduler=scheduler,
                model_parameters=model.parameters(),
            )

        return model, optim, scheduler
