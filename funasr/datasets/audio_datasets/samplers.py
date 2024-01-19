import torch
import numpy as np

from funasr.register import tables


@tables.register("batch_sampler_classes", "DynamicBatchLocalShuffleSampler")
class BatchSampler(torch.utils.data.BatchSampler):
    
    def __init__(self, dataset,
                 batch_type: str = "example",
                 batch_size: int = 100,
                 buffer_size: int = 30,
                 drop_last: bool = False,
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
        self.max_token_length = kwargs.get("max_token_length", 5000)
        self.shuffle_idx = np.arange(self.total_samples)
        self.shuffle = shuffle and is_training
    
    def __len__(self):
        return (self.total_samples-1) // self.batch_size + 1
    
    def set_epoch(self, epoch):
        np.random.seed(epoch)
    
    def __iter__(self):
        
        if self.shuffle:
            np.random.shuffle(self.shuffle_idx)
        
        batch = []
        max_token = 0
        num_sample = 0
        
        iter_num = (self.total_samples - 1) // self.buffer_size + 1
        # print("iter_num: ", iter_num)
        for iter in range(self.pre_idx + 1, iter_num):
            datalen_with_index = []
            for i in range(self.buffer_size):
                idx = iter * self.buffer_size + i
                if idx >= self.total_samples:
                    continue
                
                idx_map = self.shuffle_idx[idx]
                # prompt = self.dataset.indexed_dataset[idx_map]["prompt"]
                sample_len_cur = self.dataset.get_source_len(idx_map) + \
                                 self.dataset.get_target_len(idx_map)
                
                datalen_with_index.append([idx, sample_len_cur])
            
            datalen_with_index_sort = sorted(datalen_with_index, key=lambda x: x[1])
            for item in datalen_with_index_sort:
                idx, sample_len_cur_raw = item
                if sample_len_cur_raw > self.max_token_length:
                    continue
                
                max_token_cur = max(max_token, sample_len_cur_raw)
                max_token_padding = 1 + num_sample
                if self.batch_type == 'length':
                    max_token_padding *= max_token_cur
                if max_token_padding <= self.batch_size:
                    batch.append(idx)
                    max_token = max_token_cur
                    num_sample += 1
                else:
                    yield batch
                    batch = [idx]
                    max_token = sample_len_cur_raw
                    num_sample = 1

