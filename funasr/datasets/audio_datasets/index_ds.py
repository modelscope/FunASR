import json
import torch
import logging
import torch.distributed as dist

from funasr.register import tables


@tables.register("index_ds_classes", "IndexDSJsonl")
class IndexDSJsonl(torch.utils.data.Dataset):
    
    def __init__(self, path):
        super().__init__()
        
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
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1
            logging.warning("distributed is not initialized, only single shard")
        num_per_rank = total_num // world_size
        
        # rank = 0
        # import ipdb; ipdb.set_trace()
        self.contents = contents[rank * num_per_rank:(rank + 1) * num_per_rank]
    
        logging.info("in rank: {}, num of samplers: {}, total_num of samplers across ranks: {}".format(rank, len(self.contents), len(contents)))

    def __len__(self):
        return len(self.contents)
    
    def __getitem__(self, index):
        try:
            data = self.contents[index]
        except:
            print(index)
        return data
    
    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
        
        return data_dict["target_len"] if "target_len" in data_dict else 0
