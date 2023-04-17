import os
import logging
from multiprocessing import Pool

import numpy as np
import torch.distributed as dist


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
    logging.info("{}/{} samples in {} are filtered because of the mismatch between wav.scp and text".format(len(wav_lines),
                                                                                                     filter_count,
                                                                                                     dataset))


def calc_shape_core(root_path, frontend_conf, speech_length_min, speech_length_max, idx):
    wav_scp_file = os.path.join(root_path, "wav.scp.{}".format(idx))
    shape_file = os.path.join(root_path, "speech_shape.{}".format(idx))
    with open(wav_scp_file) as f:
        lines = f.readlines()
    with open(shape_file, "w") as f:
        for line in lines:
            sample_name, wav_path = line.strip().split()
            n_frames, feature_dim, speech_length = wav2num_frame(wav_path, frontend_conf)
            write_flag = True
            if speech_length_min > 0 and speech_length < speech_length_min:
                write_flag = False
            if speech_length_max > 0 and speech_length > speech_length_max:
                write_flag = False
            if write_flag:
                f.write("{} {},{}\n".format(sample_name, str(int(np.ceil(n_frames))), str(int(feature_dim))))
                f.flush()


def calc_shape(args, dataset, nj=32):
    shape_path = os.path.join(args.data_dir, dataset, "speech_shape")
    if os.path.exists(shape_path):
        print('Shape file for small dataset already exists.')
        return

    split_shape_path = os.path.join(args.data_dir, dataset, "shape_files")
    if os.path
    os.makedirs(split_shape_path, exist_ok=True)

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
        p.apply_async(calc_shape_core,
                      args=(shape_path, frontend_conf, speech_length_min, speech_length_max, str(i + 1)))
    print('Generating shape files, please wait a few minutes...')
    p.close()
    p.join()

    # combine
    file = os.path.join(data_dir, dataset, "speech_shape")
    with open(file, "w") as f:
        for i in range(nj):
            job_file = os.path.join(shape_path, "speech_shape.{}".format(str(i + 1)))
            with open(job_file) as job_f:
                lines = job_f.readlines()
                f.writelines(lines)
    print('Generating shape files done.')


def prepare_data(args, distributed_option):
    distributed = distributed_option.distributed
    if not distributed or distributed_option.dist_rank == 0:
        filter_wav_text(args.data_dir, args.train_set)
        filter_wav_text(args.data_dir, args.dev_set)
        dist.barrier()

        if args.dataset_type == "small" and args.train_shape_file is None:
