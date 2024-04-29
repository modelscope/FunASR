import os
import json
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import concurrent.futures
import librosa
import torch.distributed as dist


def gen_scp_from_jsonl(jsonl_file, data_type_list, wav_scp_file, text_file):

    wav_f = open(wav_scp_file, "w")
    text_f = open(text_file, "w")
    with open(jsonl_file, encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line.strip())

            prompt = data.get("prompt", "<ASR>")
            source = data[data_type_list[0]]
            target = data[data_type_list[1]]
            source_len = data.get("source_len", 1)
            target_len = data.get("target_len", 0)
            if "aishell" in source:
                target = target.replace(" ", "")
            key = data["key"]
            wav_f.write(f"{key}\t{source}\n")
            wav_f.flush()
            text_f.write(f"{key}\t{target}\n")
            text_f.flush()

    wav_f.close()
    text_f.close()


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):

    kwargs = OmegaConf.to_container(cfg, resolve=True)
    print(kwargs)

    scp_file_list = kwargs.get(
        "scp_file_list",
        ("/Users/zhifu/funasr1.0/test_local/wav.scp", "/Users/zhifu/funasr1.0/test_local/text.txt"),
    )
    if isinstance(scp_file_list, str):
        scp_file_list = eval(scp_file_list)
    data_type_list = kwargs.get("data_type_list", ("source", "target"))
    jsonl_file = kwargs.get(
        "jsonl_file_in", "/Users/zhifu/funasr1.0/test_local/audio_datasets.jsonl"
    )
    gen_scp_from_jsonl(jsonl_file, data_type_list, *scp_file_list)


"""
python -m funasr.datasets.audio_datasets.json2scp \
++scp_file_list='["/Users/zhifu/funasr1.0/test_local/wav.scp", "/Users/zhifu/funasr1.0/test_local/text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_in=/Users/zhifu/funasr1.0/test_local/audio_datasets.jsonl
"""

if __name__ == "__main__":
    main_hydra()
