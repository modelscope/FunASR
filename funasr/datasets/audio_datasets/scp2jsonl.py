import os
import json
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import concurrent.futures
import librosa
import torch.distributed as dist
from tqdm import tqdm


def gen_jsonl_from_wav_text_list(
    path, data_type_list=("source", "target"), jsonl_file_out: str = None, **kwargs
):
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    except:
        rank = 0
        world_size = 1

    cpu_cores = os.cpu_count() or 1
    print(f"convert wav.scp text to jsonl, ncpu: {cpu_cores}")
    if rank == 0:
        json_dict = {}
        for data_type, data_file in zip(data_type_list, path):
            json_dict[data_type] = {}
            with open(data_file, "r") as f:

                data_file_lists = f.readlines()
                lines_for_each_th = (len(data_file_lists) - 1) // cpu_cores + 1
                task_num = cpu_cores if len(data_file_lists) > cpu_cores else 1
                # import pdb;pdb.set_trace()
                if task_num > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:

                        futures = [
                            executor.submit(
                                parse_context_length,
                                data_file_lists[
                                    i * lines_for_each_th : (i + 1) * lines_for_each_th
                                ],
                                data_type,
                                i,
                            )
                            for i in range(task_num)
                        ]

                        for future in concurrent.futures.as_completed(futures):

                            json_dict[data_type].update(future.result())
                else:
                    res = parse_context_length(data_file_lists, data_type)
                    json_dict[data_type].update(res)

        with open(jsonl_file_out, "w") as f:
            for key in json_dict[data_type_list[0]].keys():
                jsonl_line = {"key": key}
                for data_file in data_type_list:
                    jsonl_line.update(json_dict[data_file][key])
                jsonl_line = json.dumps(jsonl_line, ensure_ascii=False)
                f.write(jsonl_line + "\n")
                f.flush()
        print(f"processed {len(json_dict[data_type_list[0]])} samples")

    else:
        pass

    if world_size > 1:
        dist.barrier()


def parse_context_length(data_list: list, data_type: str, id=0):
    pbar = tqdm(total=len(data_list), dynamic_ncols=True)
    res = {}
    for i, line in enumerate(data_list):
        pbar.update(1)
        pbar.set_description(f"cpu: {id}")
        lines = line.strip().split(maxsplit=1)
        key = lines[0]
        line = lines[1] if len(lines) > 1 else ""
        line = line.strip()
        if os.path.exists(line):
            waveform, _ = librosa.load(line, sr=16000)
            sample_num = len(waveform)
            context_len = int(sample_num / 16000 * 1000 / 10)
        else:
            context_len = len(line.split()) if " " in line else len(line)
        res[key] = {data_type: line, f"{data_type}_len": context_len}
    return res


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
    jsonl_file_out = kwargs.get(
        "jsonl_file_out", "/Users/zhifu/funasr1.0/test_local/audio_datasets.jsonl"
    )
    gen_jsonl_from_wav_text_list(
        scp_file_list, data_type_list=data_type_list, jsonl_file_out=jsonl_file_out
    )


"""
python -m funasr.datasets.audio_datasets.scp2jsonl \
++scp_file_list='["/Users/zhifu/funasr1.0/test_local/wav.scp", "/Users/zhifu/funasr1.0/test_local/text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out=/Users/zhifu/funasr1.0/test_local/audio_datasets.jsonl
"""

if __name__ == "__main__":
    main_hydra()
