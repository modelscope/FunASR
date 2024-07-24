import os
import json
import torch
import logging
import hydra
import re
from omegaconf import DictConfig, OmegaConf
import concurrent.futures
import librosa
import torch.distributed as dist
from tqdm import tqdm


def gen_jsonl_from_wav_text_list(
    path, data_type_list=("source", "target"), jsonl_file_out: str = None, model_dir: str = "iic/SenseVoiceSmall", **kwargs
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

        if "text_language" not in data_type_list or "emo_target" not in data_type_list or "event_target" not in data_type_list:
            from funasr import AutoModel

            model = AutoModel(
                model=model_dir,
            )

            rich_dict = {}
            for key in json_dict["source"].keys():
                input_wav = json_dict["source"][key]["source"]
                res = model.generate(
                input=input_wav,
                cache={},
                language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                use_itn=True,
                )
                text = res[0]["text"]
                pattern = r"<\|[^|]+\|>"
                matches = re.findall(pattern, text)
                text_language, emo_target, event_target = matches[:3]
                rich_dict[key] = [text_language, emo_target, event_target]


            if "text_language" not in data_type_list:
                data_type_list.append("text_language")
                if "text_language" not in json_dict:
                    json_dict["text_language"] = {}
                for key in json_dict["source"].keys():
                    json_dict["text_language"][key] = {}
                    json_dict["text_language"][key]["text_language"] = rich_dict[key][0]

            if "emo_target" not in data_type_list:
                data_type_list.append("emo_target")
                if "emo_target" not in json_dict:
                    json_dict["emo_target"] = {}
                for key in json_dict["source"].keys():
                    json_dict["emo_target"][key] = {}
                    json_dict["emo_target"][key]["emo_target"] = rich_dict[key][1]

            if "event_target" not in data_type_list:
                data_type_list.append("event_target")
                if "event_target" not in json_dict:
                    json_dict["event_target"] = {}
                for key in json_dict["source"].keys():
                    json_dict["event_target"][key] = {}
                    json_dict["event_target"][key]["event_target"] = rich_dict[key][2]

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

def contains_punctuation(s):
    pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
    return re.search(pattern, s) is not None

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
        if data_type == "source":
            res[key] = {data_type: line, f"{data_type}_len": context_len}
        elif data_type == "target":
            punc = contains_punctuation(line)
            if punc:
                with_or_wo_itn = "<|withitn|>"
            else:
                with_or_wo_itn = "<|woitn|>"
            res[key] = {data_type: line, f"{data_type}_len": context_len, "with_or_wo_itn": with_or_wo_itn}
        else:
            res[key] = {data_type: line}
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
    model_dir = kwargs.get("model_dir", "iic/SenseVoiceSmall")
    gen_jsonl_from_wav_text_list(
        scp_file_list, data_type_list=data_type_list, jsonl_file_out=jsonl_file_out, model_dir=model_dir
    )


"""
python -m funasr.datasets.audio_datasets.sensevoice2jsonl \
++scp_file_list='["/Users/zhifu/funasr1.0/test_local/wav.scp", "/Users/zhifu/funasr1.0/test_local/text.txt", "/Users/zhifu/funasr1.0/test_local/text_language.txt", "/Users/zhifu/funasr1.0/test_local/emo_target.txt", "/Users/zhifu/funasr1.0/test_local/event_target.txt"]' \
++data_type_list='["source", "target", "text_language", "emo_target", "event_target"]' \
++jsonl_file_out='/Users/zhifu/funasr1.0/test_local/audio_datasets.jsonl' \
++model_dir='iic/SenseVoiceSmall'
"""

if __name__ == "__main__":
    main_hydra()
