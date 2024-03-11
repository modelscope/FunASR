import os
import json
import torch
import logging
import concurrent.futures
import librosa
import torch.distributed as dist

from funasr.register import tables


@tables.register("index_ds_classes", "IndexDSJsonlRankSplit")
class IndexDSJsonlRankSplit(torch.utils.data.Dataset):
    
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

@tables.register("index_ds_classes", "IndexDSJsonl")
@tables.register("index_ds_classes", "IndexDSJsonlRankFull")
class IndexDSJsonlRankFull(torch.utils.data.Dataset):
    
    def __init__(self, path: str, **kwargs):
        super().__init__()
        
        if isinstance(path, (list, tuple)): # wav.scp, text.txt/text.trans
            from funasr.datasets.audio_datasets.scp2jsonl import gen_jsonl_from_wav_text_list
            jsonl_outdir = os.path.dirname(path[0])
            jsonl_name = "datalist_train.jsonl" if kwargs.get("is_training", True) else "datalist_val.jsonl"
            jsonl_file_out = os.path.join(jsonl_outdir, jsonl_name)
            if not os.path.exists(jsonl_file_out):
                print(f"datalist is: {path}, generate jsonl from it")
                gen_jsonl_from_wav_text_list(path, jsonl_file_out=jsonl_file_out, **kwargs)
            path = jsonl_file_out

        contents = []
        with open(path, encoding='utf-8') as fin:
            for line in fin:
                data = json.loads(line.strip())
                if "text" in data:  # for sft
                    self.contents.append(data['text'])
                if "source" in data:  # for speech lab pretrain
                    prompt = data.get("prompt", "<ASR>")
                    source = data["source"]
                    target = data["target"]
                    source_len = data.get("source_len", 1)
                    target_len = data.get("target_len", 0)
                    if "aishell" in source:
                        target = target.replace(" ", "")
                    contents.append({"source": source,
                                     "prompt": prompt,
                                     "target": target,
                                     "source_len": source_len,
                                     "target_len": target_len,
                                     }
                                    )

        self.contents = contents
        
        logging.info(
            "total_num of samplers across ranks: {}".format(len(self.contents)))
    
    def __len__(self):
        return len(self.contents)
    
    def __getitem__(self, index):
        try:
            data = self.contents[index]
        except:
            print(index)
        return data
    
    def get_source_len(self, data_dict):
        return data_dict.get("source_len", 1)
    
    def get_target_len(self, data_dict):
        
        return data_dict.get("target_len", 0)
