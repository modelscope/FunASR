import os
import time
import torch
import logging
from tqdm import tqdm
import torch.distributed as dist
from contextlib import nullcontext
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from pathlib import Path

from funasr.train_utils.device_funcs import to_device
from funasr.train_utils.recursive_op import recursive_average
from funasr.train_utils.average_nbest_models import average_checkpoints

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
    
    def __init__(self, model,
                 optim,
                 scheduler,
                 dataloader_train,
                 dataloader_val,
                 local_rank,
                 use_ddp=False,
                 use_fsdp=False,
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
        
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.output_dir = output_dir
        self.resume = kwargs.get('resume', True)
        self.start_epoch = 0
        self.max_epoch = kwargs.get('max_epoch', 100)
        self.local_rank = local_rank
        self.use_ddp = use_ddp
        self.use_fsdp = use_fsdp
        self.device = next(model.parameters()).device
        self.avg_nbest_model = kwargs.get("avg_nbest_model", 5)
        self.kwargs = kwargs
        
    
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1
            logging.warning("distributed is not initialized, only single shard")
        self.rank = rank
        self.world_size = world_size
        
        os.makedirs(os.path.join(self.output_dir, "tensorboard"), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.output_dir, "tensorboard")) if rank == 0 else None
        
    
    def _save_checkpoint(self, epoch):
        """
        Saves a checkpoint containing the model's state, the optimizer's state,
        and the scheduler's state at the end of the given epoch. This method is
        intended to be called at the end of each epoch to save the training progress.

        Args:
            epoch (int): The epoch number at which the checkpoint is being saved.
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        # Create output directory if it does not exist
        os.makedirs(self.output_dir, exist_ok=True)
        filename = os.path.join(self.output_dir, f'model.pt.ep{epoch}')
        torch.save(state, filename)
        
        print(f'Checkpoint saved to {filename}')
        latest = Path(os.path.join(self.output_dir, f'model.pt'))
        try:
            latest.unlink()
        except:
            pass

        latest.symlink_to(filename)
    
    def _resume_checkpoint(self, resume_path):
        """
        Resumes training from a checkpoint at the given file path.
        Loads the model's state, the optimizer's state, and the scheduler's state.

        Args:
            resume_path (str): The file path to the checkpoint to resume from.
        """
        ckpt = os.path.join(resume_path, "model.pt")
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Checkpoint loaded successfully from '{ckpt}'")
        else:
            print(f"No checkpoint found at '{ckpt}', starting from scratch")

        if self.use_ddp or self.use_fsdp:
            dist.barrier()
        
    def run(self):
        """
        Starts the training process, iterating over epochs, training the model,
        and saving checkpoints at the end of each epoch.
        """
        if self.resume:
            self._resume_checkpoint(self.output_dir)
        
        for epoch in range(self.start_epoch, self.max_epoch + 1):
            
            self._train_epoch(epoch)


            
            if self.use_ddp or self.use_fsdp:
                dist.barrier()
                
            self._validate_epoch(epoch)

            if self.use_ddp or self.use_fsdp:
                dist.barrier()
           
           
            if self.rank == 0:
                self._save_checkpoint(epoch)
            
            if self.use_ddp or self.use_fsdp:
                dist.barrier()
            
            self.scheduler.step()


        if self.rank == 0:
            average_checkpoints(self.output_dir, self.avg_nbest_model)
            
        if self.use_ddp or self.use_fsdp:
            dist.barrier()


        if self.writer:
            self.writer.close()
        
    
    def _train_epoch(self, epoch):
        """
        Defines the training process for a single epoch with gradient accumulation.
        Args:
            epoch (int): The current epoch number.
        """
        self.model.train()
        pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch + 1}", total=len(self.dataloader_train),
                    dynamic_ncols=True)
        
        # Set the number of steps for gradient accumulation
        accum_grad = self.kwargs.get("accum_grad", 1)
        # Initialize the gradient accumulation
        self.optim.zero_grad()
        speed_stats = {}
        time5 = time.perf_counter()
        for batch_idx, batch in enumerate(self.dataloader_train):
            time1 = time.perf_counter()
            speed_stats["data_load"] = f"{time1-time5:0.3f}"

            batch = to_device(batch, self.device)
            
            my_context = self.model.no_sync if batch_idx % accum_grad != 0 else nullcontext
            with my_context():
                time2 = time.perf_counter()
                retval = self.model(**batch)
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
                loss.backward()
                time4 = time.perf_counter()
                speed_stats["backward_time"] = f"{time4 - time3:0.3f}"
            
            # Perform an optimizer step only after accumulating enough gradients
            if (batch_idx + 1) % accum_grad == 0 or (batch_idx + 1) == len(self.dataloader_train):
                # Perform gradient clipping if it is set
                if self.kwargs.get("grad_clip", None) is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.kwargs.get("grad_clip", 10.0),
                        norm_type=self.kwargs.get("grad_clip_type", 2.0),
                    )
                    if not torch.isfinite(grad_norm):
                        logging.warning(
                            f"The grad norm is {grad_norm}. Skipping updating the model."
                        )
                        self.optim.zero_grad()  # Reset gradients
                        continue
                
                # Execute an optimization step (update model parameters)
                if self.use_ddp or self.use_fsdp:
                    dist.barrier()
                self.optim.step()
                self.scheduler.step()
                # Clear gradients for the next accumulation stage
                self.optim.zero_grad()
                total_time = f"{time.perf_counter() - time5:0.3f}"
                time5 = time.perf_counter()
                speed_stats["optim_time"] = f"{time5 - time4:0.3f}"
    
                speed_stats["total_time"] = total_time


            pbar.update(1)
            if self.local_rank == 0:
                description = (
                    f"Train epoch: {epoch}/{self.max_epoch}, "
                    f"step {batch_idx}/{len(self.dataloader_train)}, "
                    f"{speed_stats}, "
                    f"(loss: {loss.detach().cpu().item():.3f}), "
                    f"{[(k, round(v.cpu().item(), 3)) for k, v in stats.items()]}"
                )
                pbar.set_description(description)
                if self.writer:
                    self.writer.add_scalar('Loss/train', loss.item(),
                                           epoch*len(self.dataloader_train) + batch_idx)
                    for key, var in stats.items():
                        self.writer.add_scalar(f'{key}/train', var.item(),
                                               epoch * len(self.dataloader_train) + batch_idx)
                    for key, var in speed_stats.items():
                        self.writer.add_scalar(f'{key}/train', eval(var),
                                               epoch * len(self.dataloader_train) + batch_idx)
                    
            # if batch_idx == 2:
            #     break
        pbar.close()

    def _validate_epoch(self, epoch):
        """
        Defines the validation process for a single epoch.
        Should be implemented with the actual model validation steps.
    
        Args:
            epoch (int): The current epoch number.
        """
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(colour="red", desc=f"Training Epoch: {epoch + 1}", total=len(self.dataloader_val),
                        dynamic_ncols=True)
            speed_stats = {}
            time5 = time.perf_counter()
            for batch_idx, batch in enumerate(self.dataloader_val):
                time1 = time.perf_counter()
                speed_stats["data_load"] = f"{time1 - time5:0.3f}"
                batch = to_device(batch, self.device)
                time2 = time.perf_counter()
                retval = self.model(**batch)
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

                pbar.update(1)
                if self.local_rank == 0:
                    description = (
                        f"validation epoch: {epoch}/{self.max_epoch}, "
                        f"step {batch_idx}/{len(self.dataloader_train)}, "
                        f"{speed_stats}, "
                        f"(loss: {loss.detach().cpu().item():.3f}), "
                        f"{[(k, round(v.cpu().item(), 3)) for k, v in stats.items()]}"
                    )
                    pbar.set_description(description)
                    if self.writer:
                        self.writer.add_scalar('Loss/val', loss.item(),
                                               epoch*len(self.dataloader_train) + batch_idx)
                        for key, var in stats.items():
                            self.writer.add_scalar(f'{key}/val', var.item(),
                                                   epoch * len(self.dataloader_train) + batch_idx)
                        for key, var in speed_stats.items():
                            self.writer.add_scalar(f'{key}/val', eval(var),
                                                   epoch * len(self.dataloader_train) + batch_idx)