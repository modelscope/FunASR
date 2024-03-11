import torch
import numpy as np
import logging
import math
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.utils.data import BatchSampler, Sampler
import torch.distributed as dist

from funasr.register import tables


@tables.register("batch_sampler_classes", "RankFullGlobalShuffleBatchSampler")
class RankFullGlobalShuffleBatchSampler(torch.utils.data.BatchSampler):
    
    def __init__(self, dataset,
                 batch_type: str = "example",
                 batch_size: int = 100,
                 buffer_size: int = 30,
                 drop_last: bool = True,
                 shuffle: bool = True,
                 is_training: bool = True,
                 **kwargs):
        
        self.drop_last = drop_last
        self.pre_idx = -1
        self.dataset = dataset
        self.total_samples = len(dataset)
        self.batch_type = batch_type
        self.batch_size = int(batch_size)
        self.buffer_size = buffer_size
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.shuffle_idx = np.arange(self.total_samples)
        self.shuffle = shuffle and is_training
        self.length_scale_source = kwargs.get("length_scale_source", 1.0)
        
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1
        self.rank = rank
        self.world_size = world_size
        
    def __len__(self):
        return (self.total_samples - 1) // (self.batch_size * self.world_size) + 1
    
    def set_epoch(self, epoch):
        np.random.seed(epoch)
    
    def __iter__(self):
    
        batch_size_total = self.batch_size * self.world_size
        
        if self.shuffle:
            np.random.shuffle(self.shuffle_idx)
        
        batch = []
        max_token = 0
        num_sample = 0
        
        iter_num = (self.total_samples - 1) // self.buffer_size + 1
        # print("iter_num: ", iter_num)
        for iter in range(self.pre_idx + 1, iter_num):
            # if iter == iter_num -1 and self.drop_last:
            #     continue
            datalen_with_index = []
            for i in range(self.buffer_size):
                idx = iter * self.buffer_size + i
                if idx >= self.total_samples:
                    continue
                
                idx_map = self.shuffle_idx[idx]
                # prompt = self.dataset.indexed_dataset[idx_map]["prompt"]
                
                source_len = self.dataset.get_source_len(idx_map) / self.length_scale_source
                target_len = self.dataset.get_target_len(idx_map) if self.batch_type == 'length' else 0.0
                sample_len_cur = source_len + target_len
                
                datalen_with_index.append([idx, sample_len_cur])
            
            datalen_with_index_sort = sorted(datalen_with_index, key=lambda x: x[1])
            for item in datalen_with_index_sort:
                idx, sample_len_cur_raw = item
                if sample_len_cur_raw > self.max_token_length:
                    continue

                max_token_cur = max(max_token, sample_len_cur_raw)
                max_token_padding = 1 + num_sample
                # if self.batch_type != 'example':
                #     max_token_padding *= max_token_cur
                if max_token_padding <= batch_size_total:
                    batch.append(idx)
                    max_token = max_token_cur
                    num_sample += 1
                else:
                    batch_rank = batch[self.rank*self.batch_size: (self.rank+1)*self.batch_size]
                    yield batch_rank
                    batch = [idx]
                    max_token = sample_len_cur_raw
                    num_sample = 1

@tables.register("batch_sampler_classes", "DistributedSamplerWarp")
class DistributedSamplerWarp(BatchSampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True, drop_last=False):
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
            self.dataset,
            num_replicas=self.num_replicas,
            rank=self.rank,
            shuffle=self.shuffle
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


def CustomDistributedBatchSampler_fn(dataset, **kwargs):
    dataloader_args = {"dataset": dataset}
    dataloader_args["batch_sampler"] = CustomDistributedBatchSampler(dataset, **kwargs)
    dataloader_args["num_workers"] = kwargs.get("num_workers", 4)
    dataloader_args["pin_memory"] = kwargs.get("pin_memory", True)
    
    return dataloader_args

@tables.register("batch_sampler_classes", "CustomDistributedBatchSampler")
class CustomDistributedBatchSampler(Sampler):
    def __init__(self, dataset,
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
            self.total_size = (len(self.dataset) // (batch_size * num_replicas)) * (batch_size * num_replicas)
        else:
            self.total_size = math.ceil(len(self.dataset) / (batch_size * num_replicas)) * (batch_size * num_replicas)
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
            indices += (indices * (padding_size // len(indices)) + indices[:padding_size % len(indices)])

        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
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
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        # Drop the last batch if it's not full and drop_last is True
        if self.drop_last and len(batches[-1]) != self.batch_size:
            batches = batches[:-1]

        return iter(batches)

    def __len__(self):

        return self.num_samples // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch
