import torch
import numpy as np
import logging
import math
import random
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.utils.data import BatchSampler, Sampler
import torch.distributed as dist

from funasr.register import tables


@tables.register("batch_sampler_classes", "BatchSampler")
@tables.register("batch_sampler_classes", "CustomDistributedBatchSampler")
@tables.register("batch_sampler_classes", "CustomDistributedDynamicBatchSampler")
@tables.register("batch_sampler_classes", "DynamicBatchLocalShuffleSampler")
@tables.register("batch_sampler_classes", "RankFullLocalShuffleBatchSampler")
@tables.register("batch_sampler_classes", "RankFullLocalShuffleDynamicBatchSampler")
def CustomDistributedBatchSampler_fn(dataset, **kwargs):
    dataloader_args = {}
    batch_type = kwargs.get("batch_type", "example")
    if batch_type == "example":
        batch_sampler = CustomDistributedBatchSampler(dataset, **kwargs)

    else:
        if kwargs.get("sort_size", -1) > 0:
            batch_sampler = CustomDistributedBufferDynamicBatchSampler(dataset, **kwargs)
        else:
            batch_sampler = CustomDistributedDynamicBatchSampler(dataset, **kwargs)
        # batch_sampler = CustomDistributedDynamicBatchSampler(dataset, **kwargs)

    dataloader_args["batch_sampler"] = batch_sampler
    dataloader_args["num_workers"] = kwargs.get("num_workers", 4)
    dataloader_args["pin_memory"] = kwargs.get("pin_memory", True)

    return dataloader_args


class CustomDistributedBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=True,
        drop_last=False,
        is_training: bool = True,
        **kwargs,
    ):

        try:
            rank = dist.get_rank()
            num_replicas = dist.get_world_size()
        except:
            rank = 0
            num_replicas = 1
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.batch_size = batch_size
        self.is_training = is_training
        self.shuffle = shuffle and is_training
        self.drop_last = drop_last
        # self.total_size = len(dataset)
        if self.drop_last:
            self.total_size = (len(self.dataset) // (batch_size * num_replicas)) * (
                batch_size * num_replicas
            )
        else:
            self.total_size = math.ceil(len(self.dataset) / (batch_size * num_replicas)) * (
                batch_size * num_replicas
            )
        self.num_samples = int(self.total_size // self.num_replicas)
        self.epoch = 0
        self.max_token_length = kwargs.get("max_token_length", None)
        self.length_scale_source = kwargs.get("length_scale_source", 1.0)

    def __iter__(self):
        # Generate a list of indices
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (
                indices * (padding_size // len(indices)) + indices[: padding_size % len(indices)]
            )

        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # Filter out indices with length greater than the max length, if provided
        if self.max_token_length is not None:
            filtered_indices = []
            for idx in indices:
                source_len = self.dataset.get_source_len(idx) / self.length_scale_source
                if source_len <= self.max_token_length:
                    filtered_indices.append(idx)
            indices = filtered_indices

        # Now that we have only the indices for this replica, chunk them into batches
        batches = [
            indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)
        ]

        # Drop the last batch if it's not full and drop_last is True
        if self.drop_last and len(batches[-1]) != self.batch_size:
            batches = batches[:-1]

        return iter(batches)

    def __len__(self):

        return self.num_samples // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch


class CustomDistributedBufferBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=True,
        drop_last=False,
        is_training: bool = True,
        sort_size: int = 1024,
        **kwargs,
    ):

        try:
            rank = dist.get_rank()
            num_replicas = dist.get_world_size()
        except:
            rank = 0
            num_replicas = 1
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.batch_size = batch_size
        self.is_training = is_training
        self.shuffle = shuffle and is_training
        self.drop_last = drop_last
        # self.total_size = len(dataset)
        if self.drop_last:
            self.total_size = (len(self.dataset) // (batch_size * num_replicas)) * (
                batch_size * num_replicas
            )
        else:
            self.total_size = math.ceil(len(self.dataset) / (batch_size * num_replicas)) * (
                batch_size * num_replicas
            )
        self.num_samples = int(self.total_size // self.num_replicas)
        self.epoch = 0
        self.max_token_length = kwargs.get("max_token_length", None)
        self.length_scale_source = kwargs.get("length_scale_source", 1.0)
        self.sort_size = sort_size

    def __iter__(self):
        # Generate a list of indices
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (
                indices * (padding_size // len(indices)) + indices[: padding_size % len(indices)]
            )

        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # Filter out indices with length greater than the max length, if provided
        if self.max_token_length is not None:
            filtered_indices = []
            for idx in indices:
                source_len = self.dataset.get_source_len(idx) / self.length_scale_source
                if source_len <= self.max_token_length:
                    filtered_indices.append(idx)
            indices = filtered_indices

        # Buffer sorting logic
        sorted_batches = []
        buffer = []

        for idx in indices:
            buffer.append(idx)
            if len(buffer) >= self.sort_size:
                # Sort the buffer based on some criteria, e.g., dataset sample length
                buffer.sort(key=lambda x: self.dataset.get_source_len(x))
                sorted_batches.extend(self._create_batches_from_buffer(buffer))
                buffer = []

        # Handle the remaining items in the buffer
        if buffer:
            buffer.sort(key=lambda x: self.dataset.get_source_len(x))
            sorted_batches.extend(self._create_batches_from_buffer(buffer))

        return iter(sorted_batches)

    def _create_batches_from_buffer(self, buffer):
        # Function to convert the sorted buffer into batches
        batched_buffer = [
            buffer[i : i + self.batch_size] for i in range(0, len(buffer), self.batch_size)
        ]
        if self.drop_last and len(batched_buffer[-1]) != self.batch_size:
            batched_buffer = batched_buffer[:-1]
        return batched_buffer

    def __len__(self):

        return self.num_samples // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch


class CustomDistributedDynamicBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=True,
        drop_last=False,
        is_training: bool = True,
        **kwargs,
    ):

        try:
            rank = dist.get_rank()
            num_replicas = dist.get_world_size()
        except:
            rank = 0
            num_replicas = 1
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.batch_size = batch_size
        self.is_training = is_training
        self.shuffle = shuffle and is_training
        self.drop_last = drop_last

        self.total_size = len(self.dataset)
        # self.num_samples = int(math.ceil(self.total_size / self.num_replicas))
        self.epoch = 0
        self.max_token_length = kwargs.get("max_token_length", 2048)
        self.length_scale_source = kwargs.get("length_scale_source", 1.0)

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices = indices[self.rank : self.total_size : self.num_replicas]

        batches = []
        batch = []
        max_len_in_batch = 0
        current_batch_length = 0

        for idx in indices:
            sample_length = self.dataset.get_source_len(idx)
            if sample_length > self.max_token_length:
                continue
            potential_batch_length = (
                max_len_in_batch if sample_length < max_len_in_batch else sample_length
            ) * (len(batch) + 1)

            if potential_batch_length <= self.batch_size:
                batch.append(idx)
                if sample_length > max_len_in_batch:
                    max_len_in_batch = sample_length
                    # current_batch_length = max_len_in_batch * len(batch)
            else:
                batches.append(batch)
                batch = [idx]
                max_len_in_batch = sample_length
                # current_batch_length = max_len_in_batch

        # Add the last batch if it's not empty and we're not dropping it
        if batch and (not self.drop_last or len(batch) * max_len_in_batch == self.batch_size):
            batches.append(batch)

        return iter(batches)

    def __len__(self):

        return 1

    def set_epoch(self, epoch):
        self.epoch = epoch


class CustomDistributedBufferDynamicBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        batch_type="token",
        num_replicas=None,
        rank=None,
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
        self.length_scale_source = kwargs.get("length_scale_source", 1.0)
        self.batch_size_sample_max = kwargs.get("batch_size_sample_max", 200)
        self.start_step = start_step
        self.batch_num = 1
        if self.start_step > 0:
            logging.info(f"Warning, start_step > 0, dataloader start from step: {self.start_step}")
        # super().__init__(
        #     dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last
        # )

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            random.seed(self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Create sorted buffers and form batches
        buffer_batches = []
        for i in range(0, len(indices), self.sort_size):
            buffer = sorted(
                indices[i : i + self.sort_size], key=lambda idx: self.dataset.get_source_len(idx)
            )
            batch = []
            max_len_in_batch = 0
            count = 1
            for idx in buffer:
                original_sample_length = self.dataset.get_source_len(idx)
                if original_sample_length > self.max_token_length:
                    continue
                sample_length = 1 if self.batch_type == "example" else original_sample_length
                potential_batch_length = max(max_len_in_batch, sample_length) * (len(batch) + 1)
                if potential_batch_length <= self.batch_size and count < self.batch_size_sample_max:
                    batch.append(idx)
                    max_len_in_batch = max(max_len_in_batch, sample_length)
                    count += 1
                else:
                    buffer_batches.append(batch)
                    batch = [idx]
                    max_len_in_batch = sample_length
                    count = 1
            if batch:
                buffer_batches.append(batch)

        # Ensure each rank gets the same number of batches, duplicate data if needed
        batches_per_rank = math.ceil(len(buffer_batches) / self.num_replicas)
        total_batches_needed = batches_per_rank * self.num_replicas

        extra_batches = total_batches_needed - len(buffer_batches)
        buffer_batches += random.choices(buffer_batches, k=extra_batches)

        # Evenly distribute batches from buffer_batches to each rank
        rank_batches = [[] for _ in range(self.num_replicas)]
        for i, batch in enumerate(buffer_batches):
            rank_batches[i % self.num_replicas].append(batch)

        # Assign all batches for the current rank directly
        final_batches = rank_batches[self.rank][self.start_step :]
        self.batch_num = len(final_batches)

        logging.info(
            f"rank: {self.rank}, dataloader start from step: {self.start_step}, batch_num: {len(rank_batches[self.rank])}, after: {self.batch_num}"
        )
        return iter(final_batches)

    def __len__(self):
        # Calculate the number of batches per epoch for the current rank
        return self.batch_num

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSamplerWarp(BatchSampler):
    def __init__(
        self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True, drop_last=False
    ):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Create an instance of the DistributedSampler
        self.sampler = DistributedSampler(
            self.dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle
        )

        # Call BatchSampler's constructor
        super().__init__(self.sampler, batch_size, drop_last)

    def __iter__(self):
        # If we shuffle, we need to call the set_epoch method
        if self.shuffle:
            self.sampler.set_epoch(self.epoch)

        # Generate batch indices using the parent class
        return super().__iter__()

    def set_epoch(self, epoch):
        self.epoch = epoch
