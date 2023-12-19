import argparse
import logging
import os
import sys
from io import BytesIO
from collections.abc import Sequence
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from funasr.train_utils.set_all_random_seed import set_all_random_seed
from funasr.models.lora.utils import mark_only_lora_as_trainable
from funasr.optimizers import optim_classes
from funasr.schedulers import scheduler_classes
from funasr.train_utils.load_pretrained_model import load_pretrained_model
from funasr.train_utils.initialize import initialize
# from funasr.tokenizer.build_tokenizer import build_tokenizer
# from funasr.tokenizer.token_id_converter import TokenIDConverter
# from funasr.tokenizer.funtoken import build_tokenizer
from funasr.train_utils.trainer import Trainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from funasr.download.download_from_hub import download_model
from funasr.utils.register import registry_tables

@hydra.main(config_name=None, version_base=None)
def main_hydra(kwargs: DictConfig):
	import pdb; pdb.set_trace()
	assert "model" in kwargs
	if "model_conf" not in kwargs:
		logging.info("download models from model hub: {}".format(kwargs.get("model_hub", "ms")))
		kwargs = download_model(is_training=kwargs.get("is_training", True), **kwargs)
	

	main(**kwargs)


def main(**kwargs):
	# preprocess_config(kwargs)
	# import pdb; pdb.set_trace()
	# set random seed
	registry_tables.print_register_tables()
	set_all_random_seed(kwargs.get("seed", 0))
	torch.backends.cudnn.enabled = kwargs.get("cudnn_enabled", torch.backends.cudnn.enabled)
	torch.backends.cudnn.benchmark = kwargs.get("cudnn_benchmark", torch.backends.cudnn.benchmark)
	torch.backends.cudnn.deterministic = kwargs.get("cudnn_deterministic", True)
	
	local_rank = int(os.environ.get('LOCAL_RANK', 0))
	# Check if we are using DDP or FSDP
	use_ddp = 'WORLD_SIZE' in os.environ and int(os.environ["WORLD_SIZE"]) > 1
	use_fsdp = kwargs.get("use_fsdp", None)
	if use_ddp or use_fsdp:
		dist.init_process_group(backend=kwargs.get("backend", "nccl"), init_method='env://')
		torch.cuda.set_device(local_rank)
	
	# save config.yaml
	if (use_ddp or use_fsdp) and dist.get_rank() == 0 or not (use_ddp or use_fsdp) and local_rank == 0:
		os.makedirs(kwargs.get("output_dir", "./"), exist_ok=True)
		yaml_file = os.path.join(kwargs.get("output_dir", "./"), "config.yaml")
		OmegaConf.save(config=kwargs, f=yaml_file)
		logging.info("config.yaml is saved to: %s", yaml_file)

	tokenizer = kwargs.get("tokenizer", None)
	if tokenizer is not None:
		tokenizer_class = registry_tables.tokenizer_classes.get(tokenizer.lower())
		tokenizer = tokenizer_class(**kwargs["tokenizer_conf"])
		kwargs["tokenizer"] = tokenizer
	
	# build frontend if frontend is none None
	frontend = kwargs.get("frontend", None)
	if frontend is not None:
		frontend_class = registry_tables.frontend_classes.get(frontend.lower())
		frontend = frontend_class(**kwargs["frontend_conf"])
		kwargs["frontend"] = frontend
	
	# import pdb;
	# pdb.set_trace()
	# build model
	model_class = registry_tables.model_classes.get(kwargs["model"].lower())
	model = model_class(**kwargs, **kwargs["model_conf"], vocab_size=len(tokenizer.token_list))



	# init_param
	init_param = kwargs.get("init_param", None)
	if init_param is not None:
		if not isinstance(init_param, (list, tuple)):
			init_param = (init_param,)
		logging.info("init_param is not None: %s", init_param)
		for p in init_param:
			logging.info(f"Loading pretrained params from {p}")
			load_pretrained_model(
				model=model,
				init_param=p,
				ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
				oss_bucket=kwargs.get("oss_bucket", None),
			)
	else:
		initialize(model, kwargs.get("init", "kaiming_normal"))


	# freeze_param
	freeze_param = kwargs.get("freeze_param", None)
	if freeze_param is not None:
		freeze_param = eval(freeze_param)
		if isinstance(freeze_param, Sequence):
			freeze_param = (freeze_param,)
		logging.info("freeze_param is not None: %s", freeze_param)
		for t in freeze_param:
			for k, p in model.named_parameters():
				if k.startswith(t + ".") or k == t:
					logging.info(f"Setting {k}.requires_grad = False")
					p.requires_grad = False
	

	if use_ddp:
		model = model.cuda(local_rank)
		model = DDP(model, device_ids=[local_rank],
		            find_unused_parameters=kwargs.get("train_conf", {}).get("find_unused_parameters", False))
	elif use_fsdp:
		model = FSDP(model).cuda(local_rank)
	else:
		model = model.to(device=kwargs.get("device", "cuda"))
		
		
	# optim
	optim = kwargs.get("optim", "adam")
	assert optim in optim_classes
	optim_class = optim_classes.get(optim)
	optim = optim_class(model.parameters(), **kwargs.get("optim_conf"))
	
	# scheduler
	scheduler = kwargs.get("scheduler", "warmuplr")
	assert scheduler in scheduler_classes
	scheduler_class = scheduler_classes.get(scheduler)
	scheduler = scheduler_class(optim, **kwargs.get("scheduler_conf"))

	# import pdb;
	# pdb.set_trace()
	# dataset
	dataset_class = registry_tables.dataset_classes.get(kwargs.get("dataset", "AudioDataset").lower())
	dataset_tr = dataset_class(kwargs.get("train_data_set_list"), frontend=frontend, tokenizer=tokenizer, **kwargs.get("dataset_conf"))

	# dataloader
	batch_sampler = kwargs["dataset_conf"].get("batch_sampler", "DynamicBatchLocalShuffleSampler")
	batch_sampler_class = registry_tables.batch_sampler_classes.get(batch_sampler.lower())
	batch_sampler = batch_sampler_class(dataset_tr, **kwargs.get("dataset_conf"))
	dataloader_tr = torch.utils.data.DataLoader(dataset_tr,
	                                            collate_fn=dataset_tr.collator,
	                                            batch_sampler=batch_sampler,
	                                            num_workers=kwargs.get("dataset_conf").get("num_workers", 4),
	                                            pin_memory=True)
	

	
	trainer = Trainer(
	    model=model,
	    optim=optim,
	    scheduler=scheduler,
	    dataloader_train=dataloader_tr,
	    dataloader_val=None,
		local_rank=local_rank,
		use_ddp=use_ddp,
		use_fsdp=use_fsdp,
		**kwargs.get("train_conf"),
	)
	trainer.run()
	
	if use_ddp or use_fsdp:
		torch.distributed.destroy_process_group()

	

if __name__ == "__main__":
	main_hydra()