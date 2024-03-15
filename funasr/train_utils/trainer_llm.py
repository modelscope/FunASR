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
    
    def __init__(self,
                 local_rank,
                 use_ddp: bool = False,
                 use_fsdp: bool = False,
                 use_fp16: bool = False,
                 output_dir: str="./",
                 **kwargs):
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
        self.resume = kwargs.get('resume', True)
        self.start_epoch = 0
        self.max_epoch = kwargs.get('max_epoch', 100)
        self.local_rank = local_rank
        self.use_ddp = use_ddp
        self.use_fsdp = use_fsdp
        self.device = kwargs.get('device', "cuda")
        self.avg_nbest_model = kwargs.get("avg_nbest_model", 5)
        # self.kwargs = kwargs
        self.log_interval = kwargs.get("log_interval", 50)
        self.batch_total = 0
        self.use_fp16 = use_fp16
        self.disable_gpu_cache = kwargs.get("disable_gpu_cache", True)
        # scaler = GradScaler(enabled=use_fp16) if use_fp16 else None
        # scaler = ShardedGradScaler(enabled=use_fp16) if use_fsdp else scaler
        # self.scaler = scaler
        self.save_checkpoint_interval = kwargs.get("save_checkpoint_interval", 5000)
        self.accum_grad = kwargs.get("accum_grad", 1)
        self.grad_clip = kwargs.get("grad_clip", 10.0)
        self.grad_clip_type = kwargs.get("grad_clip_type", 2.0)
        self.validate_interval = kwargs.get("validate_interval", 5000)
        
    
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1
            logging.warning("distributed is not initialized, only single shard")
        self.rank = rank
        self.world_size = world_size
        

        
    
    def save_checkpoint(self, epoch,
                        step=None,
                        model=None,
                        optim=None,
                        scheduler=None,
                        scaler=None,
                        ):
        """
        Saves a checkpoint containing the model's state, the optimizer's state,
        and the scheduler's state at the end of the given epoch. This method is
        intended to be called at the end of each epoch to save the training progress.

        Args:
            epoch (int): The epoch number at which the checkpoint is being saved.
        """
        if self.rank == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            if scaler:
                state["scaler_state"] = scaler.state_dict()
            # Create output directory if it does not exist
            os.makedirs(self.output_dir, exist_ok=True)
            if step is None:
                filename = os.path.join(self.output_dir, f'model.pt.ep{epoch}')
            else:
                filename = os.path.join(self.output_dir, f'model.pt.ep{epoch}.{step}')
            
            torch.save(state, filename)
            
            print(f'\nCheckpoint saved to {filename}\n')
            latest = Path(os.path.join(self.output_dir, f'model.pt'))
            torch.save(state, latest)
        
        if self.use_ddp or self.use_fsdp:
            dist.barrier()
    
    def resume_checkpoint(self,
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
                checkpoint = torch.load(ckpt)
                self.start_epoch = checkpoint['epoch'] + 1
                # self.model.load_state_dict(checkpoint['state_dict'])
                src_state = checkpoint['state_dict']
                dst_state = model.state_dict()
                for k in dst_state.keys():
                    if not k.startswith("module.") and "module."+k in src_state.keys():
                        k_ddp = "module."+k
                    else:
                        k_ddp = k
                    if k_ddp in src_state.keys():
                        dst_state[k] = src_state[k_ddp]
                    else:
                        print(f"Miss key in ckpt: model: {k}, ckpt: {k_ddp}")
    
                model.load_state_dict(dst_state)
                optim.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                if scaler is not None and 'scaler_state' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state'])
                print(f"Checkpoint loaded successfully from '{ckpt}'")
            else:
                print(f"No checkpoint found at '{ckpt}', does not resume status!")
    
        if self.use_ddp or self.use_fsdp:
            dist.barrier()
        
    # def train(self):
    #     """
    #     Starts the training process, iterating over epochs, training the model,
    #     and saving checkpoints at the end of each epoch.
    #     """
    #     if self.resume:
    #         self.resume_checkpoint(self.output_dir)
    #
    #     for epoch in range(self.start_epoch, self.max_epoch + 1):
    #         time1 = time.perf_counter()
    #         self.train_epoch(epoch)
    #
    #
    #
    #         if self.use_ddp or self.use_fsdp:
    #             dist.barrier()
    #
    #         self._validate_epoch(epoch)
    #
    #         if self.use_ddp or self.use_fsdp:
    #             dist.barrier()
    #
    #
    #         if self.rank == 0:
    #             self._save_checkpoint(epoch)
    #
    #         if self.use_ddp or self.use_fsdp:
    #             dist.barrier()
    #
    #         self.scheduler.step()
    #
    #         time2 = time.perf_counter()
    #         time_escaped = (time2 - time1)/3600.0
    #         print(f"\nrank: {self.local_rank}, time_escaped_epoch: {time_escaped:.3f} hours, estimated to finish {self.max_epoch} epoch: {(self.max_epoch-epoch)*time_escaped:.3f} hours\n")
    #
    #     if self.rank == 0:
    #         average_checkpoints(self.output_dir, self.avg_nbest_model)
    #
    #     if self.use_ddp or self.use_fsdp:
    #         dist.barrier()
    #
    #
    #     if writer:
    #         writer.close()
    #
    
    def train_epoch(self,
                model=None,
                optim=None,
                scheduler=None,
                scaler=None,
                dataloader_train=None,
                dataloader_val=None,
                epoch=None,
                writer=None,
                    ):
        """
        Defines the training process for a single epoch with gradient accumulation.
        Args:
            epoch (int): The current epoch number.
        """
        model.train()

        
        # Set the number of steps for gradient accumulation
        accum_grad = self.accum_grad
        # Initialize the gradient accumulation
        optim.zero_grad()
        speed_stats = {}
        time5 = time.perf_counter()
        
        for batch_idx, batch in enumerate(dataloader_train):
            self.batch_total += 1
            time1 = time.perf_counter()
            speed_stats["data_load"] = f"{time1-time5:0.3f}"

            batch = to_device(batch, self.device)
            
            my_context = model.no_sync if batch_idx % accum_grad != 0 else nullcontext
            with my_context():
                time2 = time.perf_counter()
                with maybe_autocast(self.use_fp16):
                    retval = model(**batch)
                    
                if self.disable_gpu_cache: torch.cuda.empty_cache()

                time3 = time.perf_counter()
                speed_stats["forward_time"] = f"{time3 - time2:0.3f}"
                loss, stats, weight = retval
                stats = {k: v for k, v in stats.items() if v is not None}
                if self.use_ddp or self.use_fsdp:
                    # Apply weighted averaging for loss and stats
                    loss = (loss * weight.type(loss.dtype)).sum()
                    # if distributed, this method can also apply all_reduce()
                    stats, weight = recursive_average(stats, weight, distributed=True)
                    # Now weight is summation over all workers
                    loss /= weight
                    # Multiply world_size because DistributedDataParallel
                    # automatically normalizes the gradient by world_size.
                    loss *= self.world_size
                # Scale the loss since we're not updating for every mini-batch
                loss = loss / accum_grad
                if self.use_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                time4 = time.perf_counter()
                speed_stats["backward_time"] = f"{time4 - time3:0.3f}"
            
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
                total_time = f"{time.perf_counter() - time5:0.3f}"
                time5 = time.perf_counter()
                speed_stats["optim_time"] = f"{time5 - time4:0.3f}"
    
                speed_stats["total_time"] = total_time
                lr = scheduler.get_last_lr()[0]

                self.log(epoch, batch_idx,
                         batch_num_epoch=len(dataloader_train),
                         lr=lr,
                         loss=loss.detach().cpu().item(),
                         speed_stats=speed_stats,
                         stats=stats,
                         writer=writer,
                         tag="train",
                         )

            if (batch_idx + 1) % self.validate_interval == 0:
                self.validate_epoch(
                    model=model,
                    dataloader_val=dataloader_val,
                    epoch=epoch,
                    writer=writer
                )

            if (batch_idx+1) % self.save_checkpoint_interval == 0 and self.rank == 0:
                self.save_checkpoint(epoch, model=model, optim=optim, scheduler=scheduler, scaler=scaler, step=batch_idx+1)

        
        if self.use_ddp or self.use_fsdp:
            dist.barrier()
        
        

    def validate_epoch(self,
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
        model.eval()
        
        with torch.no_grad():
            
            speed_stats = {}
            time5 = time.perf_counter()
            for batch_idx, batch in enumerate(dataloader_val):
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
                    stats, weight = recursive_average(stats, weight, distributed=True)
                    # Now weight is summation over all workers
                    loss /= weight
                    # Multiply world_size because DistributedDataParallel
                    # automatically normalizes the gradient by world_size.
                    loss *= self.world_size
                # Scale the loss since we're not updating for every mini-batch
                loss = loss
                time4 = time.perf_counter()

                
                self.log(epoch, batch_idx,
                         batch_num_epoch=len(dataloader_val),
                         lr=0.0,
                         loss=loss.detach().cpu().item(),
                         speed_stats=speed_stats,
                         stats=stats,
                         writer=writer,
                         tag="train",
                         )

        model.train()
        
        
    def log(self,
            epoch=0,
            batch_idx=0,
            batch_num_epoch=-1,
            lr=0.0,
            loss=0.0,
            speed_stats=None,
            stats=None,
            writer=None,
            tag="train",
            ):
        
        if (batch_idx + 1) % self.log_interval == 0:
            
            gpu_info = "GPU, memory: {:.3f} GB, " \
                       "{:.3f} GB, " \
                       "{:.3f} GB, " \
                       "{:.3f} GB".format(torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
                                          torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
                                          torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
                                          torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
                                          )
            
            time_now = datetime.now()
            time_now = time_now.strftime("%Y-%m-%d %H:%M:%S")
            description = (
                f"{time_now}, "
                f"rank: {self.local_rank}, "
                f"epoch: {epoch}/{self.max_epoch}, "
                f"step: {batch_idx + 1}/{batch_num_epoch}, total step: {self.batch_total}, "
                f"(loss: {loss:.3f}), "
                f"(lr: {lr:.3e}), "
                f"{[(k, round(v.cpu().item(), 3)) for k, v in stats.items()]}, "
                f"{speed_stats}, "
                f"{gpu_info}"
            )
            logging.info(description)
            
            if writer is not None:
                writer.add_scalar(f'rank{self.local_rank}_Loss/{tag}', loss, self.batch_total)
                writer.add_scalar(f'rank{self.local_rank}_lr/{tag}', lr, self.batch_total)
                for key, var in stats.items():
                    writer.add_scalar(f'rank{self.local_rank}_{key}/{tag}', var.item(), self.batch_total)
                for key, var in speed_stats.items():
                    writer.add_scalar(f'rank{self.local_rank}_{key}/{tag}', eval(var), self.batch_total)
        
    def close(self, writer=None):
        if writer is not None:
            writer.close()
    
        if self.use_ddp or self.use_fsdp:
            torch.distributed.destroy_process_group()