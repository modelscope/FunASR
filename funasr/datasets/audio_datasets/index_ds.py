import os
import json
import torch
import logging

import librosa
import random
import torch.distributed as dist

from funasr.register import tables


@tables.register("index_ds_classes", "IndexDSJsonl")
@tables.register("index_ds_classes", "IndexDSJsonlRankFull")
@tables.register("index_ds_classes", "IndexDSJsonlRankSplit")
class IndexDSJsonlRankFull(torch.utils.data.Dataset):

    def __init__(self, path: str, **kwargs):
        super().__init__()
        self.max_source_length = kwargs.get("max_source_length", 2048)
        self.min_source_length = kwargs.get("min_source_length", 0)
        self.max_target_length = kwargs.get("max_target_length", 2048)
        self.min_target_length = kwargs.get("min_target_length", 0)
        self.max_token_length = kwargs.get("max_token_length", 2200)

        is_training = kwargs.get("is_training", True)
        if not (path.endswith(".jsonl") or path.endswith(".json")):
            # jsonl list file
            data_split_num = kwargs.get("data_split_num", 1)
            data_split_i = kwargs.get("data_split_i", 0)

            if not is_training:
                data_split_num = 1
                data_split_i = 0
            with open(path, encoding="utf-8") as fin:
                file_list_all = fin.readlines()

                num_per_slice = (len(file_list_all) - 1) // data_split_num + 1  # 16
                file_list = file_list_all[
                    data_split_i * num_per_slice : (data_split_i + 1) * num_per_slice
                ]
                logging.info(
                    f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list: {file_list}, \nfile_list_all: {file_list_all}"
                )

        else:
            file_list = [path]

        # total_num = len(file_list)
        # try:
        #     rank = dist.get_rank()
        #     world_size = dist.get_world_size()
        # except:
        #     rank = 0
        #     world_size = 1
        #     logging.info("distributed is not initialized, only single shard")
        #
        # if not kwargs.get("rank_split", False):
        #     logging.info(f"Warning, rank_split disenabled, batch and shuffle data in global")
        #     rank = 0
        #     world_size = 1
        #
        # num_per_rank = total_num // world_size
        # if num_per_rank * world_size < total_num:
        #     logging.info(f"Warning, jsonl file:{total_num} could not be divided by world_size: {world_size}, {path}")
        #     total_num_needed = num_per_rank * world_size
        #
        #     extra_num = total_num_needed - total_num
        #     file_list_tmp = random.choices(file_list, k=extra_num)
        #     file_list += file_list_tmp
        #     logging.info(f"Warning, after random choices: {file_list}")
        #
        # file_list_rank = file_list[rank * num_per_rank:(rank + 1) * num_per_rank]
        #
        # logging.info(
        #     f"is_training: {is_training}, file_list_rank: {file_list_rank}")

        # contents = []
        # for file_json in file_list_rank:
        contents = []
        for file_json in file_list:
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    data = json.loads(line.strip())
                    if "text" in data:  # for sft
                        contents.append(data["text"])
                    if "source" in data:  # for speech lab pretrain
                        prompt = data.get("prompt", "<ASR>")
                        source = data["source"].replace(
                            "/cpfs01", "/cpfs_speech/data"
                        )  # only use in alibaba gpu group: .replace("/cpfs01", "/cpfs_speech/data")
                        target = data["target"]
                        source_len = data.get("source_len", 1)
                        target_len = data.get("target_len", 0)
                        if "aishell" in source:
                            target = target.replace(" ", "")
                        if (
                            source_len < self.min_source_length
                            or source_len > self.max_source_length
                        ):
                            continue
                        if (
                            target_len < self.min_target_length
                            or target_len > self.max_target_length
                        ):
                            continue

                        if (source_len + target_len) > self.max_token_length:
                            continue

                        contents_i = {
                            "source": source,
                            "prompt": prompt,
                            "target": target,
                            "source_len": source_len,
                            "target_len": target_len,
                        }
                        text_language = data.get("text_language", None)
                        if text_language is not None:
                            contents_i["text_language"] = text_language
                        # audio_language = data.get("audio_language", None)
                        # if audio_language is not None:
                        #     contents_i["audio_language"] = audio_language
                        contents.append(contents_i)

        self.contents = contents

        logging.info("total_num of samplers: {}, {}".format(len(self.contents), path))

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):

        data = self.contents[index]

        return data

    def get_source_len(self, data_dict):
        return data_dict.get("source_len", 1)

    def get_target_len(self, data_dict):

        return data_dict.get("target_len", 0)
