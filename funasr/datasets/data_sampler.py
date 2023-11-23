import torch

import numpy as np

class BatchSampler(torch.utils.data.BatchSampler):
	
	def __init__(self, dataset, batch_size_type: str="example", batch_size: int=100, sort_size: int=30, drop_last: bool=False, shuffle: bool=True, **kwargs):
		
		self.drop_last = drop_last
		self.pre_idx = -1
		self.dataset = dataset
		self.total_samples = len(dataset)
		# self.batch_size_type = args.batch_size_type
		# self.batch_size = args.batch_size
		# self.sort_size = args.sort_size
		# self.max_length_token = args.max_length_token
		self.batch_size_type = batch_size_type
		self.batch_size = batch_size
		self.sort_size = sort_size
		self.max_length_token = kwargs.get("max_length_token", 5000)
		self.shuffle_idx = np.arange(self.total_samples)
		self.shuffle = shuffle

	
	def __len__(self):
		return self.total_samples

	def __iter__(self):
		print("in sampler")
		
		if self.shuffle:
			np.random.shuffle(self.shuffle_idx)
			
		batch = []
		max_token = 0
		num_sample = 0

		iter_num = (self.total_samples-1) // self.sort_size + 1
		print("iter_num: ", iter_num)
		for iter in range(self.pre_idx + 1, iter_num):
			datalen_with_index = []
			for i in range(self.sort_size):
				idx = iter * self.sort_size + i
				if idx >= self.total_samples:
					continue

				idx_map = self.shuffle_idx[idx]
				# prompt = self.dataset.indexed_dataset[idx_map]["prompt"]
				sample_len_cur = self.dataset.indexed_dataset[idx_map]["source_len"] + \
				                 self.dataset.indexed_dataset[idx_map]["target_len"]

				datalen_with_index.append([idx, sample_len_cur])
			
			datalen_with_index_sort = sorted(datalen_with_index, key=lambda x: x[1])
			for item in datalen_with_index_sort:
				idx, sample_len_cur_raw = item
				if sample_len_cur_raw > self.max_length_token:
					continue

				max_token_cur = max(max_token, sample_len_cur_raw)
				max_token_padding = 1 + num_sample
				if self.batch_size_type == 'token':
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
					
		