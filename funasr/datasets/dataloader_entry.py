
import logging
import torch

from funasr.register import tables

@tables.register("dataloader_classes", "DataloaderMapStyle")
def DataloaderMapStyle(frontend=None, tokenizer=None, **kwargs):
	# dataset
	logging.info("Build dataloader")
	dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "AudioDataset"))
	dataset_tr = dataset_class(kwargs.get("train_data_set_list"), frontend=frontend, tokenizer=tokenizer, is_training=True, **kwargs.get("dataset_conf"))
	dataset_val = dataset_class(kwargs.get("valid_data_set_list"), frontend=frontend, tokenizer=tokenizer, is_training=False, **kwargs.get("dataset_conf"))
	
	# dataloader
	batch_sampler = kwargs["dataset_conf"].get("batch_sampler", "DynamicBatchLocalShuffleSampler")
	batch_sampler_val = None
	if batch_sampler is not None:
		batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
		batch_sampler = batch_sampler_class(dataset_tr, **kwargs.get("dataset_conf"))
		batch_sampler_val = batch_sampler_class(dataset_val, is_training=False, **kwargs.get("dataset_conf"))
	
	dataloader_tr = torch.utils.data.DataLoader(dataset_tr, collate_fn=dataset_tr.collator, **batch_sampler)
	dataloader_val = torch.utils.data.DataLoader(dataset_val, collate_fn=dataset_val.collator, **batch_sampler_val)
	
	return dataloader_tr, dataloader_val


@tables.register("dataloader_classes", "DataloaderIterable")
def DataloaderIterable(frontend=None, tokenizer=None, **kwargs):
	logging.info("Build dataloader")
	dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "LargeDataset"))
	dataset_tr = dataset_class(kwargs.get("train_data_set_list"), frontend=frontend, tokenizer=tokenizer,
	                           is_training=True, **kwargs.get("dataset_conf"))
	dataset_val = dataset_class(kwargs.get("valid_data_set_list"), frontend=frontend, tokenizer=tokenizer,
	                            is_training=False, **kwargs.get("dataset_conf"))
	
	return dataset_tr, dataset_val