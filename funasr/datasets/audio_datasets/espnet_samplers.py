import torch
import numpy as np
import logging
import math
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.utils.data import BatchSampler, Sampler
import torch.distributed as dist
import random
from funasr.register import tables


@tables.register("batch_sampler_classes", "EspnetStyleBatchSampler")
def EspnetStyleBatchSampler_fn(dataset, **kwargs):
    dataloader_args = {}

    batch_sampler = EspnetStyleBatchSampler(dataset, **kwargs)
    dataloader_args["batch_sampler"] = batch_sampler
    dataloader_args["num_workers"] = kwargs.get("num_workers", 4)
    dataloader_args["pin_memory"] = kwargs.get("pin_memory", True)
    
    return dataloader_args


import torch
from torch.utils.data import Dataset, DistributedSampler
import math
import random


class EspnetStyleBatchSampler(DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None,
                 shuffle=True, drop_last=False, max_sample_length=2000,
                 batch_type="length"):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, drop_last=drop_last)
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_sample_length = max_sample_length
        self.batch_type = batch_type
    
    def __iter__(self):
        # Get the list of indices of all samples
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        
        # Shuffle the indices if required
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(dataset_size, generator=g).tolist()
        
        # Sort indices by sample length
        sorted_indices = sorted(indices, key=lambda idx: self.dataset.get_source_len(idx))
        
        # Organize batches based on 'length' or 'example'
        buffer_batches = []
        batch = []
        max_len_in_batch = 0  # Tracks the max sample length within the current batch
        
        for idx in sorted_indices:
            original_sample_length = self.dataset.get_source_len(idx)
            if original_sample_length > self.max_sample_length:  # Skip samples that exceed the max length
                continue
            # Set sample_length based on the batch type
            sample_length = 1 if self.batch_type == "example" else original_sample_length
            # Calculate potential batch size with the new sample
            potential_batch_length = max(max_len_in_batch, sample_length) * (len(batch) + 1)
            # Add index to batch if it doesn't exceed batch size limit
            if potential_batch_length <= self.batch_size:
                batch.append(idx)
                max_len_in_batch = max(max_len_in_batch, sample_length)
            else:
                # Save the current batch and start a new one
                buffer_batches.append(batch)
                batch = [idx]
                max_len_in_batch = sample_length
        
        # Add the last batch if it shouldn't be dropped
        if batch and not self.drop_last:
            buffer_batches.append(batch)
        
        # Shuffle the list of batches
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(buffer_batches)
        
        # Ensure each rank gets the same number of batches
        batches_per_rank = math.ceil(len(buffer_batches) / self.num_replicas)
        total_batches_needed = batches_per_rank * self.num_replicas
        extra_batches = total_batches_needed - len(buffer_batches)
        # Add extra batches by random selection, if needed
        buffer_batches += random.choices(buffer_batches, k=extra_batches)
        
        # Allocate the batches to the current rank
        start_idx = self.rank * batches_per_rank
        end_idx = start_idx + batches_per_rank
        rank_batches = buffer_batches[start_idx:end_idx]
        
        # Return an iterator over the batches for the current rank
        return iter(rank_batches)
    
    def __len__(self):
        # Calculate the number of batches per epoch for the current rank
        return math.ceil(len(self.dataset) / self.batch_size / self.num_replicas)
    
    def set_epoch(self, epoch):
        # Set the epoch for shuffling
        self.epoch = epoch


