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
    def __init__(
        self,
        dataset,
        batch_size,
        batch_type="token",
        rank=None,
        num_replicas=None,
        rank_split=False,
        shuffle=True,
        drop_last=False,
        is_training: bool = True,
        sort_size: int = 1024,
        start_step: int = 0,
        **kwargs,
    ):

        try:
            rank = dist.get_rank()
            num_replicas = dist.get_world_size()
        except:
            rank = 0
            num_replicas = 1
        # if rank_split:
        #     logging.info(f"Warning, rank_split: {rank_split}, batch and shuffle data in local rank")
        #     rank = 0
        #     num_replicas = 1
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.is_training = is_training
        self.shuffle = shuffle and is_training
        self.drop_last = drop_last

        self.total_size = len(self.dataset)
        self.num_samples = int(math.ceil(self.total_size / self.num_replicas))
        self.epoch = 0
        self.sort_size = sort_size * num_replicas
        self.max_token_length = kwargs.get("max_token_length", 2048)
        self.min_token_length = kwargs.get("min_token_length", 0)
        self.length_scale_source = kwargs.get("length_scale_source", 1.0)
        self.start_step = start_step
        self.batch_num = 1
        if self.start_step > 0:
            logging.info(f"Warning, start_step > 0, dataloader start from step: {self.start_step}")
        # super().__init__(dataset, num_replicas=num_replicas, rank=rank,
        #                  shuffle=shuffle, drop_last=drop_last)

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            random.seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Sort indices by sample length
        sorted_indices = sorted(indices, key=lambda idx: self.dataset.get_source_len(idx))

        # Organize batches based on 'length' or 'example'
        buffer_batches = []
        batch = []
        max_len_in_batch = 0  # Tracks the max sample length within the current batch

        for idx in sorted_indices:

            # original_sample_length = self.dataset.get_source_len(idx)
            # if (
            #     original_sample_length < self.min_token_length
            #     or original_sample_length > self.max_token_length
            # ):  # Skip samples that exceed the max length
            #     continue

            # sample_length = 1 if self.batch_type == "example" else original_sample_length

            # Set sample_length based on the batch type
            if self.batch_type == "example":
                sample_length = 1
            elif self.batch_type == "token":
                sample_length = self.dataset.get_source_len(idx) + int(
                    self.dataset.get_target_len(idx) * 1.2
                )
            else:
                sample_length = self.dataset.get_source_len(idx)
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
        if batch and (not self.drop_last or len(batch) * max_len_in_batch == self.batch_size):
            buffer_batches.append(batch)

        # Shuffle the list of batches
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(buffer_batches)

        # Ensure each rank gets the same number of batches
        batches_per_rank = int(math.ceil(len(buffer_batches) / self.num_replicas))
        total_batches_needed = batches_per_rank * self.num_replicas
        extra_batches = total_batches_needed - len(buffer_batches)
        # Add extra batches by random selection, if needed
        buffer_batches += random.choices(buffer_batches, k=extra_batches)

        # Allocate the batches to the current rank
        start_idx = self.rank * batches_per_rank
        end_idx = start_idx + batches_per_rank
        rank_batches = buffer_batches[start_idx + self.start_step : end_idx]

        self.batch_num = len(rank_batches)

        logging.info(
            f"rank: {self.rank}, dataloader start from step: {self.start_step}, batch_num: {end_idx-start_idx}, batch_num_after_step: {len(rank_batches)}"
        )
        # Return an iterator over the batches for the current rank
        return iter(rank_batches)

    def __len__(self):
        # Calculate the number of batches per epoch for the current rank
        return self.batch_num

    def set_epoch(self, epoch):
        # Set the epoch for shuffling
        self.epoch = epoch
