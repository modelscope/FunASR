import os
import json
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import concurrent.futures
import librosa
import torch.distributed as dist
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def gen_scp_from_jsonl(jsonl_file, jsonl_file_out, ncpu):
    jsonl_file_out_f = open(jsonl_file_out, "w")
    with open(jsonl_file, encoding="utf-8") as fin:
        lines = fin.readlines()

        num_total = len(lines)
        if ncpu > 1:
            # 使用ThreadPoolExecutor限制并发线程数
            with ThreadPoolExecutor(max_workers=ncpu) as executor:
                # 提交任务到线程池
                futures = {executor.submit(update_data, lines, i) for i in tqdm(range(num_total))}

                # 等待所有任务完成，这会阻塞直到所有提交的任务完成
                for future in concurrent.futures.as_completed(futures):
                    # 这里可以添加额外的逻辑来处理完成的任务，但在这个例子中我们只是等待
                    pass
        else:
            for i in range(num_total):
                update_data(lines, i)
        logging.info("All audio durations have been processed.")

        for line in lines:

            jsonl_file_out_f.write(line + "\n")
            jsonl_file_out_f.flush()

    jsonl_file_out_f.close()


def update_data(lines, i):
    line = lines[i]
    data = json.loads(line.strip())

    wav_path = data["source"].replace("/cpfs01", "/cpfs_speech/data")
    if os.path.exists(wav_path):
        waveform, _ = librosa.load(wav_path, sr=16000)
        sample_num = len(waveform)
        source_len = int(sample_num / 16000 * 1000 / 10)
        source_len_old = data["source_len"]
        # if (source_len_old - source_len) > 100 or (source_len - source_len_old) > 100:
        #     logging.info(f"old: {source_len_old}, new: {source_len}, wav: {wav_path}")
        data["source_len"] = source_len
        data["source"] = wav_path
        jsonl_line = json.dumps(data, ensure_ascii=False)
        lines[i] = jsonl_line


def update_wav_len(jsonl_file_list_in, jsonl_file_out_dir, ncpu=1):

    os.makedirs(jsonl_file_out_dir, exist_ok=True)
    with open(jsonl_file_list_in, "r") as f:
        data_file_lists = f.readlines()

        for i, jsonl in enumerate(data_file_lists):
            filename_with_extension = os.path.basename(jsonl.strip())
            jsonl_file_out = os.path.join(jsonl_file_out_dir, filename_with_extension)
            logging.info(f"{i}/{len(data_file_lists)}, jsonl: {jsonl}, {jsonl_file_out}")

            gen_scp_from_jsonl(jsonl.strip(), jsonl_file_out, ncpu)


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):

    kwargs = OmegaConf.to_container(cfg, resolve=True)
    logging.info(kwargs)

    jsonl_file_list_in = kwargs.get(
        "jsonl_file_list_in", "/Users/zhifu/funasr1.0/data/list/data_jsonl.list"
    )
    jsonl_file_out_dir = kwargs.get("jsonl_file_out_dir", "/Users/zhifu/funasr1.0/data_tmp")
    ncpu = kwargs.get("ncpu", 1)
    update_wav_len(jsonl_file_list_in, jsonl_file_out_dir, ncpu)
    # gen_scp_from_jsonl(jsonl_file_list_in, jsonl_file_out_dir)


"""
python -m funasr.datasets.audio_datasets.json2scp \
++scp_file_list='["/Users/zhifu/funasr1.0/test_local/wav.scp", "/Users/zhifu/funasr1.0/test_local/text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_in=/Users/zhifu/funasr1.0/test_local/audio_datasets.jsonl
"""

if __name__ == "__main__":
    main_hydra()
