import torch
import json
import torch.distributed as dist

class AudioDatasetJsonl(torch.utils.data.Dataset):
	
	def __init__(self, path, data_parallel_rank=0, data_parallel_size=1):
		super().__init__()
		data_parallel_size = dist.get_world_size()
		contents = []
		with open(path, encoding='utf-8') as fin:
			for line in fin:
				data = json.loads(line.strip())
				if "text" in data:  # for sft
					self.contents.append(data['text'])
				if "source" in data:  # for speech lab pretrain
					prompt = data["prompt"]
					source = data["source"]
					target = data["target"]
					source_len = data["source_len"]
					target_len = data["target_len"]

					contents.append({"source": source,
					                 "prompt": prompt,
					                 "target": target,
					                 "source_len": source_len,
					                 "target_len": target_len,
					                 }
					                )
		
		self.contents = []
		total_num = len(contents)
		num_per_rank = total_num // data_parallel_size
		rank = dist.get_rank()
		# import ipdb; ipdb.set_trace()
		self.contents = contents[rank * num_per_rank:(rank + 1) * num_per_rank]


	def __len__(self):
		return len(self.contents)
	
	def __getitem__(self, index):
		return self.contents[index]
