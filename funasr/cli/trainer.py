import torch
import os
from funasr.torch_utils.device_funcs import to_device
import logging
from tqdm import tqdm
from contextlib import nullcontext
import torch.distributed as dist

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
		self.output_dir = kwargs.get('output_dir', './')
		self.resume = kwargs.get('resume', None)
		self.start_epoch = 1
		self.max_epoch = kwargs.get('max_epoch', 100)
		self.local_rank = local_rank
		self.use_ddp = use_ddp
		self.use_fsdp = use_fsdp
		self.device = torch.device("cuda", local_rank)
		self.kwargs = kwargs
		
		if self.resume:
			self._resume_checkpoint(self.resume)
	
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
		filename = os.path.join(self.output_dir, f'model.e{epoch}.pb')
		torch.save(state, filename)
		print(f'Checkpoint saved to {filename}')
	
	def _resume_checkpoint(self, resume_path):
		"""
		Resumes training from a checkpoint at the given file path.
		Loads the model's state, the optimizer's state, and the scheduler's state.

		Args:
			resume_path (str): The file path to the checkpoint to resume from.
		"""
		if os.path.isfile(resume_path):
			checkpoint = torch.load(resume_path)
			self.start_epoch = checkpoint['epoch'] + 1
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optim.load_state_dict(checkpoint['optimizer'])
			self.scheduler.load_state_dict(checkpoint['scheduler'])
			print(f"Checkpoint loaded successfully from '{resume_path}' at (epoch {checkpoint['epoch']})")
		else:
			print(f"No checkpoint found at '{resume_path}', starting from scratch")
		
	def run(self):
		"""
		Starts the training process, iterating over epochs, training the model,
		and saving checkpoints at the end of each epoch.
		"""
		for epoch in range(self.start_epoch, self.max_epoch + 1):
			self._train_epoch(epoch)
			# self._validate_epoch(epoch)
			if dist.get_rank() == 0:
				self._save_checkpoint(epoch)
			# self.scheduler.step()
	
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
		accumulation_steps = self.kwargs.get("accumulation_steps", 1)
		# Initialize the gradient accumulation
		self.optim.zero_grad()
		
		for batch_idx, batch in enumerate(self.dataloader_train):
			batch = to_device(batch, self.device)
			
			my_context = self.model.no_sync if batch_idx % accumulation_steps != 0 else nullcontext
			with my_context():
				retval = self.model(**batch)
				loss, stats, weight = retval
				
				# Scale the loss since we're not updating for every mini-batch
				loss = loss / accumulation_steps
				loss.backward()
			
			# Perform an optimizer step only after accumulating enough gradients
			if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.dataloader_train):
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
				self.optim.step()
				self.scheduler.step()
				# Clear gradients for the next accumulation stage
				self.optim.zero_grad()
			
			pbar.update(1)
			if self.local_rank == 0:
				pbar.set_description(
					f"Training Epoch: {epoch + 1}/{self.max_epoch}, step {batch_idx}/{len(self.dataloader_train)}  (loss: {loss.detach().float()})")
			
		pbar.close()
	
	# def _train_epoch(self, epoch):
	# 	"""
	# 	Defines the training process for a single epoch.
	# 	Should be implemented with the actual model training steps.
	#
	# 	Args:
	# 		epoch (int): The current epoch number.
	# 	"""
	# 	self.model.train()
	# 	pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch + 1}", total=len(self.dataloader_train), dynamic_ncols=True)
	# 	for batch_idx, batch in enumerate(self.dataloader_train):
	# 		batch = to_device(batch, "cpu")
	# 		retval = self.model(**batch)
	# 		loss, stats, weight = retval
	# 		self.optim.zero_grad()
	# 		loss.backward()
	#
	# 		# compute the gradient norm to check if it is normal or not
	# 		grad_norm = torch.nn.utils.clip_grad_norm_(
	# 			self.model.parameters(),
	# 			max_norm=self.kwargs.get("grad_clip", 10.0),
	# 			norm_type=self.kwargs.get("grad_clip_type", 2.0),
	# 		)
	# 		if not torch.isfinite(grad_norm):
	# 			logging.warning(
	# 				f"The grad norm is {grad_norm}. Skipping updating the model."
	# 			)
	# 			continue
	# 		self.optim.step()
	# 		self.scheduler.step()
	# 		pbar.update(1)
	# 		pbar.set_description(
	# 			f"Training Epoch: {epoch + 1}/{self.max_epoch}, step {batch_idx}/{len(self.dataloader_train)}  (loss: {loss.detach().float()})")
	#
	# 	pbar.close()
	#

	def _validate_epoch(self, epoch):
		"""
		Defines the validation process for a single epoch.
		Should be implemented with the actual model validation steps.
	
		Args:
			epoch (int): The current epoch number.
		"""
		self.model.eval()
		with torch.no_grad():
			for data, target in self.dataloader_val:
				# Implement the model validation steps here
				pass

# # Example usage
# if __name__ == "__main__":
# 	# Assuming the following objects have already been correctly created and initialized:
# 	# model, optim, scheduler, dataloader_train, and dataloader_val.
# 	trainer = Trainer(
# 	    max_epoch=10,
# 	    model=model,
# 	    optim=optim,
# 	    scheduler=scheduler,
# 	    dataloader_train=dataloader_train,
# 	    dataloader_val=dataloader_val,
# 	    output_dir='path_to_save_model',
# 	    resume='path_to_checkpoint_if_any'
# 	)
# 	trainer.run()