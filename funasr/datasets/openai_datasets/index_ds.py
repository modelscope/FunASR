import os
import json
import torch
import logging

import librosa
import random
import torch.distributed as dist

from funasr.register import tables


@tables.register("index_ds_classes", "OpenAIIndexDSJsonl")
class OpenAIIndexDSJsonl(torch.utils.data.Dataset):  # torch.utils.data.Dataset

    def __init__(self, path: str, **kwargs):
        super().__init__()

        self.max_source_length = kwargs.get("max_source_length", 3000)
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

        contents = []
        for file_json in file_list:
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    data = data_dict["messages"]
                    speech_length = data_dict.get("speech_length", -1) // 8
                    text_length = data_dict.get("text_length", 0)
                    if speech_length > self.max_source_length:
                        logging.info(
                            f"speech_length: {speech_length} > {self.max_source_length}, drop it"
                        )
                        continue
                    if text_length > self.max_target_length:
                        continue

                    self.max_target_length = kwargs.get("max_target_length", 2048)

                    system, user, assistant = [], [], []
                    for i, item in enumerate(data):
                        role = item["role"]
                        content = item["content"]
                        if role == "system":
                            system.append(content)
                        elif role == "user":
                            user.append(content)
                        elif role == "assistant":
                            if "wav_path" in item:
                                wav_path = item["wav_path"]
                                assistant.append([content, {"wav_path": wav_path}])
                            else:
                                assistant.append(content)
                    if len(system) == 0:
                        system = ["You are a helpful assistant."]
                    system = system * len(user)

                    contents_i = {
                        "system": system,
                        "user": user,
                        "assistant": assistant,
                        "source_len": speech_length + text_length,
                    }
                    if "key" in data_dict:
                        contents_i["key"] = data_dict["key"]
                    contents.append(contents_i)

        self.contents = contents

        logging.info("total_num of samplers: {}, {}".format(len(self.contents), path))

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):

        data = self.contents[index]

        return data

    def get_source_len(self, data_dict):
        source_len = data_dict.get("source_len", -1)
        if source_len < 0:
            source_len = len(data_dict["system"]) + len(data_dict["user"])
        return source_len

    def get_target_len(self, data_dict):

        return 0


@tables.register("index_ds_classes", "OpenAIIndexDSJsonlForFullDuplexVAD")
class OpenAIIndexDSJsonlForFullDuplexVAD(torch.utils.data.Dataset):  # torch.utils.data.Dataset

    def __init__(self, path: str, **kwargs):
        super().__init__()

        self.max_source_length = kwargs.get("max_source_length", 3000)
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

        contents = []
        for file_json in file_list:
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    data = data_dict["messages"]
                    for message in data:
                        if message["role"] == "user":
                            message["content"] = message["content"].replace(
                                "/home/qinglin.zql/project/dataset/gpt-4o/vad",
                                "/cpfs_speech/qinglin.zql/project/datasets/gpt-4o/vad",
                            )
                            message["content"] = message["content"].replace(
                                "/cpfs_speech/qinglin.zql/project/datasets/gpt-4o/vad/alimeeting/wav",
                                "/cpfs_speech/qinglin.zql/project/datasets/gpt-4o/vad/alimeeting/alimeeting_vad/wav",
                            )

                    speech_length = data_dict.get("speech_length", -1) // 8
                    text_length = data_dict.get("text_length", 0)
                    task = data_dict["task"]
                    last_total_time = data[-1]["end_time"] - data[-1]["start_time"]
                    if task == "turn-taking":
                        true_time_span = data[-1]["turn-taking-gap_time-added"]
                    elif task == "barge-in":
                        true_time_span = last_total_time - data[-1]["barge-in-0"]
                    if speech_length > self.max_source_length:
                        logging.info(
                            f"speech_length: {speech_length} > {self.max_source_length}, drop it"
                        )
                        continue
                    if text_length > self.max_target_length:
                        continue

                    self.max_target_length = kwargs.get("max_target_length", 2048)

                    system, user, assistant = [], [], []
                    for i, item in enumerate(data):
                        role = item["role"]
                        content = item["content"]
                        if role == "system":
                            system.append(content)
                        elif role == "user":
                            user.append(content)
                        elif role == "assistant":
                            assistant.append(content)

                    system = system * len(user)
                    assert len(user) - 1 == len(assistant)
                    assistant.append("")

                    contents_i = {
                        "system": system,
                        "user": user,
                        "assistant": assistant,
                        "source_len": speech_length + text_length,
                        "task": task,
                        "true_time_span": true_time_span,
                        "last_total_time": last_total_time,
                    }

                    contents.append(contents_i)

        self.contents = contents

        logging.info("total_num of samplers: {}, {}".format(len(self.contents), path))

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):

        data = self.contents[index]

        return data

    def get_source_len(self, data_dict):
        source_len = data_dict.get("source_len", -1)
        if source_len < 0:
            source_len = len(data_dict["system"]) + len(data_dict["user"])
        return source_len

    def get_target_len(self, data_dict):

        return 0


@tables.register("index_ds_classes", "OpenAIIndexDSJsonlMel")
class OpenAIIndexDSJsonlMel(torch.utils.data.Dataset):  # torch.utils.data.Dataset

    def __init__(self, path: str, **kwargs):
        super().__init__()

        # tts text tokenizer related
        tts_token_type = "whisper_rich_ttsfrd"
        ttsfrd_res_dir = "/nfs/neo.dzh/pip_wheels/ttsfrd/9.5.5"
        from funasr.models.llm_asr.tts_text_tokenizer.build_tokenizer import build_tokenizer

        self.tts_text_tokenizer = build_tokenizer(
            tts_token_type,
            bpemodel=ttsfrd_res_dir,
            p_word2phn=1.0,
        )

        self.max_source_length = kwargs.get("max_source_length", 3000)
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

        contents = []
        for file_json in file_list:
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    data = data_dict["messages"]
                    speech_length = data_dict.get("speech_length", -1) // 8
                    text_length = data_dict.get("text_length", 0)
                    if speech_length > self.max_source_length:
                        logging.info(
                            f"speech_length: {speech_length} > {self.max_source_length}, drop it"
                        )
                        continue
                    if text_length > self.max_target_length:
                        continue

                    self.max_target_length = kwargs.get("max_target_length", 2048)

                    system, user, assistant = [], [], []
                    for i, item in enumerate(data):
                        role = item["role"]
                        content = item["content"]
                        if role == "system":
                            system.append(content)
                        elif role == "user":
                            user.append(content)
                        elif role == "assistant":
                            if "wav_path" in item:
                                wav_path = item["wav_path"]
                                assistant.append([content, {"wav_path": wav_path}])
                            else:
                                assistant.append(content)

                    system = system * len(user)

                    for i, (system_i, user_i, assistant_i) in enumerate(
                        zip(system, user, assistant)
                    ):

                        contents_i = {
                            "system": system[: i + 1],
                            "user": user[: i + 1],
                            "assistant": assistant[: i + 1],
                            "source_len": speech_length + text_length,
                        }
                        contents.append(contents_i)

        self.contents = contents

        logging.info("total_num of samplers: {}, {}".format(len(self.contents), path))

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):

        data = self.contents[index]

        return data

    def get_source_len(self, data_dict):
        source_len = data_dict.get("source_len", -1)
        if source_len < 0:
            source_len = len(data_dict["system"]) + len(data_dict["user"])
        return source_len

    def get_target_len(self, data_dict):

        return 0


if __name__ == "__main__":
    index_ds = OpenAIIndexDSJsonl(
        path="/Users/zhifu/funasr1.0/test_local/data_tmp/tmp_wav_10.jsonl"
    )
    print(index_ds.contents)
    pass
