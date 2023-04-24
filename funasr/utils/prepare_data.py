import logging
import os
import shutil
from multiprocessing import Pool

import numpy as np
import torch.distributed as dist
import torchaudio


def filter_wav_text(data_dir, dataset):
    wav_file = os.path.join(data_dir, dataset, "wav.scp")
    text_file = os.path.join(data_dir, dataset, "text")
    with open(wav_file) as f_wav, open(text_file) as f_text:
        wav_lines = f_wav.readlines()
        text_lines = f_text.readlines()
    os.rename(wav_file, "{}.bak".format(wav_file))
    os.rename(text_file, "{}.bak".format(text_file))
    wav_dict = {}
    for line in wav_lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        wav_dict[parts[0]] = parts[1]
    text_dict = {}
    for line in text_lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        text_dict[parts[0]] = " ".join(parts[1:]).lower()
    filter_count = 0
    with open(wav_file, "w") as f_wav, open(text_file, "w") as f_text:
        for sample_name, wav_path in wav_dict.items():
            if sample_name in text_dict.keys():
                f_wav.write(sample_name + " " + wav_path + "\n")
                f_text.write(sample_name + " " + text_dict[sample_name] + "\n")
            else:
                filter_count += 1
    logging.info(
        "{}/{} samples in {} are filtered because of the mismatch between wav.scp and text".format(len(wav_lines),
                                                                                                   filter_count,
                                                                                                   dataset))


def wav2num_frame(wav_path, frontend_conf):
    waveform, sampling_rate = torchaudio.load(wav_path)
    n_frames = (waveform.shape[1] * 1000.0) / (sampling_rate * frontend_conf["frame_shift"] * frontend_conf["lfr_n"])
    feature_dim = frontend_conf["n_mels"] * frontend_conf["lfr_m"]
    return n_frames, feature_dim


def calc_shape_core(root_path, args, idx):
    wav_scp_file = os.path.join(root_path, "wav.scp.{}".format(idx))
    shape_file = os.path.join(root_path, "speech_shape.{}".format(idx))
    with open(wav_scp_file) as f:
        lines = f.readlines()
    frontend_conf = args.frontend_conf
    dataset_conf = args.dataset_conf
    speech_length_min = dataset_conf.speech_length_min if hasattr(dataset_conf, "speech_length_min") else -1
    speech_length_max = dataset_conf.speech_length_max if hasattr(dataset_conf, "speech_length_max") else -1
    with open(shape_file, "w") as f:
        for line in lines:
            sample_name, wav_path = line.strip().split()
            n_frames, feature_dim = wav2num_frame(wav_path, frontend_conf)
            write_flag = True
            if n_frames > 0 and speech_length_min > 0:
                write_flag = n_frames >= speech_length_min
            if n_frames > 0 and speech_length_max > 0:
                write_flag = n_frames <= speech_length_max
            if write_flag:
                f.write("{} {},{}\n".format(sample_name, str(int(np.ceil(n_frames))), str(int(feature_dim))))
                f.flush()


def calc_shape(args, dataset, nj=32):
    shape_path = os.path.join(args.data_dir, dataset, "speech_shape")
    if os.path.exists(shape_path):
        logging.info('Shape file for small dataset already exists.')
        return

    split_shape_path = os.path.join(args.data_dir, dataset, "shape_files")
    if os.path.exists(split_shape_path):
        shutil.rmtree(split_shape_path)
    os.mkdir(split_shape_path)

    # split
    wav_scp_file = os.path.join(args.data_dir, dataset, "wav.scp")
    with open(wav_scp_file) as f:
        lines = f.readlines()
        num_lines = len(lines)
        num_job_lines = num_lines // nj
    start = 0
    for i in range(nj):
        end = start + num_job_lines
        file = os.path.join(shape_path, "wav.scp.{}".format(str(i + 1)))
        with open(file, "w") as f:
            if i == nj - 1:
                f.writelines(lines[start:])
            else:
                f.writelines(lines[start:end])
        start = end

    p = Pool(nj)
    for i in range(nj):
        p.apply_async(calc_shape_core, args=(split_shape_path, args, str(i + 1)))
    logging.info("Generating shape files, please wait a few minutes...")
    p.close()
    p.join()

    # combine
    with open(shape_path, "w") as f:
        for i in range(nj):
            job_file = os.path.join(split_shape_path, "speech_shape.{}".format(str(i + 1)))
            with open(job_file) as job_f:
                lines = job_f.readlines()
                f.writelines(lines)
    logging.info('Generating shape files done.')


def generate_data_list(data_dir, dataset, nj=100):
    list_file = os.path.join(data_dir, dataset, "data.list")
    if os.path.exists(list_file):
        logging.info('Data list for large dataset already exists.')
        return
    split_path = os.path.join(data_dir, dataset, "split")
    if os.path.exists(split_path):
        shutil.rmtree(split_path)
    os.mkdir(split_path)

    with open(os.path.join(data_dir, dataset, "wav.scp")) as f_wav:
        wav_lines = f_wav.readlines()
    with open(os.path.join(data_dir, dataset, "text")) as f_text:
        text_lines = f_text.readlines()
    num_lines = len(wav_lines)
    num_job_lines = num_lines // nj
    start = 0
    for i in range(nj):
        end = start + num_job_lines
        split_path_nj = os.path.join(split_path, str(i + 1))
        os.mkdir(split_path_nj)
        wav_file = os.path.join(split_path_nj, "wav.scp")
        text_file = os.path.join(split_path_nj, "text")
        with open(wav_file, "w") as fw, open(text_file, "w") as ft:
            if i == nj - 1:
                fw.writelines(wav_lines[start:])
                ft.writelines(text_lines[start:])
            else:
                fw.writelines(wav_lines[start:end])
                ft.writelines(text_lines[start:end])
        start = end

    with open(list_file, "w") as f_data:
        for i in range(nj):
            wav_path = os.path.join(split_path, str(i + 1), "wav.scp")
            text_path = os.path.join(split_path, str(i + 1), "text")
            f_data.write(wav_path + " " + text_path + "\n")


def prepare_data(args, distributed_option):
    if args.dataset_type == "small" and args.train_data_path_and_name_and_type is not None:
        return
    if args.dataset_type == "large" and args.train_data_file is not None:
        return
    distributed = distributed_option.distributed
    if not distributed or distributed_option.dist_rank == 0:
        filter_wav_text(args.data_dir, args.train_set)
        filter_wav_text(args.data_dir, args.dev_set)

        if args.dataset_type == "small" and args.train_shape_file is None:
            calc_shape(args, args.train_set)
            calc_shape(args, args.dev_set)

        if args.dataset_type == "large" and args.train_data_file is None:
            generate_data_list(args.data_dir, args.train_set)
            generate_data_list(args.data_dir, args.dev_set)

    args.train_shape_file = [os.path.join(args.data_dir, args.train_set, "speech_shape")]
    args.valid_shape_file = [os.path.join(args.data_dir, args.dev_set, "speech_shape")]
    args.train_data_file = os.path.join(args.data_dir, args.train_set, "data.list")
    args.valid_data_file = os.path.join(args.data_dir, args.dev_set, "data.list")
    if distributed:
        dist.barrier()
