import logging
import os
import shutil
from multiprocessing import Pool

import kaldiio
import numpy as np
import librosa
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
        text_dict[parts[0]] = " ".join(parts[1:])
    filter_count = 0
    with open(wav_file, "w") as f_wav, open(text_file, "w") as f_text:
        for sample_name, wav_path in wav_dict.items():
            if sample_name in text_dict.keys():
                f_wav.write(sample_name + " " + wav_path + "\n")
                f_text.write(sample_name + " " + text_dict[sample_name] + "\n")
            else:
                filter_count += 1
    logging.info("{}/{} samples in {} are filtered because of the mismatch between wav.scp and text".
                 format(filter_count, len(wav_lines), dataset))


def wav2num_frame(wav_path, frontend_conf):
    try:
        waveform, sampling_rate = torchaudio.load(wav_path)
    except:
        waveform, sampling_rate = librosa.load(wav_path)
        waveform = np.expand_dims(waveform, axis=0)
    n_frames = (waveform.shape[1] * 1000.0) / (sampling_rate * frontend_conf["frame_shift"] * frontend_conf["lfr_n"])
    feature_dim = frontend_conf["n_mels"] * frontend_conf["lfr_m"]
    return n_frames, feature_dim


def calc_shape_core(root_path, args, idx):
    file_name = args.data_file_names.split(",")[0]
    data_name = args.dataset_conf.get("data_names", "speech,text").split(",")[0]
    scp_file = os.path.join(root_path, "{}.{}".format(file_name, idx))
    shape_file = os.path.join(root_path, "{}_shape.{}".format(data_name, idx))
    with open(scp_file) as f:
        lines = f.readlines()
    data_type = args.dataset_conf.get("data_types", "sound,text").split(",")[0]
    if data_type == "sound":
        frontend_conf = args.frontend_conf
        dataset_conf = args.dataset_conf
        length_min = dataset_conf.speech_length_min if hasattr(dataset_conf, "{}_length_min".format(data_name)) else -1
        length_max = dataset_conf.speech_length_max if hasattr(dataset_conf, "{}_length_max".format(data_name)) else -1
        with open(shape_file, "w") as f:
            for line in lines:
                sample_name, wav_path = line.strip().split()
                n_frames, feature_dim = wav2num_frame(wav_path, frontend_conf)
                write_flag = True
                if n_frames > 0 and length_min > 0:
                    write_flag = n_frames >= length_min
                if n_frames > 0 and length_max > 0:
                    write_flag = n_frames <= length_max
                if write_flag:
                    f.write("{} {},{}\n".format(sample_name, str(int(np.ceil(n_frames))), str(int(feature_dim))))
                    f.flush()
    elif data_type == "kaldi_ark":
        dataset_conf = args.dataset_conf
        length_min = dataset_conf.speech_length_min if hasattr(dataset_conf, "{}_length_min".format(data_name)) else -1
        length_max = dataset_conf.speech_length_max if hasattr(dataset_conf, "{}_length_max".format(data_name)) else -1
        with open(shape_file, "w") as f:
            for line in lines:
                sample_name, feature_path = line.strip().split()
                feature = kaldiio.load_mat(feature_path)
                n_frames, feature_dim = feature.shape
                write_flag = True
                if n_frames > 0 and length_min > 0:
                    write_flag = n_frames >= length_min
                if n_frames > 0 and length_max > 0:
                    write_flag = n_frames <= length_max
                if write_flag:
                    f.write("{} {},{}\n".format(sample_name, str(int(np.ceil(n_frames))), str(int(feature_dim))))
                    f.flush()
    elif data_type == "text":
        with open(shape_file, "w") as f:
            for line in lines:
                sample_name, text = line.strip().split(maxsplit=1)
                n_tokens = len(text.split())
                f.write("{} {}\n".format(sample_name, str(int(np.ceil(n_tokens)))))
                f.flush()
    else:
        raise RuntimeError("Unsupported data_type: {}".format(data_type))


def calc_shape(args, dataset, nj=64):
    data_name = args.dataset_conf.get("data_names", "speech,text").split(",")[0]
    shape_path = os.path.join(args.data_dir, dataset, "{}_shape".format(data_name))
    if os.path.exists(shape_path):
        logging.info('Shape file for small dataset already exists.')
        return

    split_shape_path = os.path.join(args.data_dir, dataset, "{}_shape_files".format(data_name))
    if os.path.exists(split_shape_path):
        shutil.rmtree(split_shape_path)
    os.mkdir(split_shape_path)

    # split
    file_name = args.data_file_names.split(",")[0]
    scp_file = os.path.join(args.data_dir, dataset, file_name)
    with open(scp_file) as f:
        lines = f.readlines()
        num_lines = len(lines)
        num_job_lines = num_lines // nj
    start = 0
    for i in range(nj):
        end = start + num_job_lines
        file = os.path.join(split_shape_path, "{}.{}".format(file_name, str(i + 1)))
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
            job_file = os.path.join(split_shape_path, "{}_shape.{}".format(data_name, str(i + 1)))
            with open(job_file) as job_f:
                lines = job_f.readlines()
                f.writelines(lines)
    logging.info('Generating shape files done.')


def generate_data_list(args, data_dir, dataset, nj=64):
    data_names = args.dataset_conf.get("data_names", "speech,text").split(",")
    file_names = args.data_file_names.split(",")
    concat_data_name = "_".join(data_names)
    list_file = os.path.join(data_dir, dataset, "{}_data.list".format(concat_data_name))
    if os.path.exists(list_file):
        logging.info('Data list for large dataset already exists.')
        return
    split_path = os.path.join(data_dir, dataset, "split")
    if os.path.exists(split_path):
        shutil.rmtree(split_path)
    os.mkdir(split_path)

    data_lines_list = []
    for file_name in file_names:
        with open(os.path.join(data_dir, dataset, file_name)) as f:
            lines = f.readlines()
            data_lines_list.append(lines)
    num_lines = len(data_lines_list[0])
    num_job_lines = num_lines // nj
    start = 0
    for i in range(nj):
        end = start + num_job_lines
        split_path_nj = os.path.join(split_path, str(i + 1))
        os.mkdir(split_path_nj)
        for file_id, file_name in enumerate(file_names):
            file = os.path.join(split_path_nj, file_name)
            with open(file, "w") as f:
                if i == nj - 1:
                    f.writelines(data_lines_list[file_id][start:])
                else:
                    f.writelines(data_lines_list[file_id][start:end])
        start = end

    with open(list_file, "w") as f_data:
        for i in range(nj):
            path = ""
            for file_name in file_names:
                path = path + " " + os.path.join(split_path, str(i + 1), file_name)
            f_data.write(path + "\n")


def prepare_data(args, distributed_option):
    data_names = args.dataset_conf.get("data_names", "speech,text").split(",")
    data_types = args.dataset_conf.get("data_types", "sound,text").split(",")
    file_names = args.data_file_names.split(",")
    batch_type = args.dataset_conf["batch_conf"]["batch_type"]
    print("data_names: {}, data_types: {}, file_names: {}".format(data_names, data_types, file_names))
    assert len(data_names) == len(data_types) == len(file_names)
    if args.dataset_type == "small":
        args.train_shape_file = [os.path.join(args.data_dir, args.train_set, "{}_shape".format(data_names[0]))]
        args.valid_shape_file = [os.path.join(args.data_dir, args.valid_set, "{}_shape".format(data_names[0]))]
        args.train_data_path_and_name_and_type, args.valid_data_path_and_name_and_type = [], []
        for file_name, data_name, data_type in zip(file_names, data_names, data_types):
            args.train_data_path_and_name_and_type.append(
                ["{}/{}/{}".format(args.data_dir, args.train_set, file_name), data_name, data_type])
            args.valid_data_path_and_name_and_type.append(
                ["{}/{}/{}".format(args.data_dir, args.valid_set, file_name), data_name, data_type])
        if os.path.exists(args.train_shape_file[0]):
            assert os.path.exists(args.valid_shape_file[0])
            print('shape file for small dataset already exists.')
            return
    else:
        concat_data_name = "_".join(data_names)
        args.train_data_file = os.path.join(args.data_dir, args.train_set, "{}_data.list".format(concat_data_name))
        args.valid_data_file = os.path.join(args.data_dir, args.valid_set, "{}_data.list".format(concat_data_name))
        if os.path.exists(args.train_data_file):
            assert os.path.exists(args.valid_data_file)
            print('data list for large dataset already exists.')
            return

    distributed = distributed_option.distributed
    if not distributed or distributed_option.dist_rank == 0:
        if hasattr(args, "filter_input") and args.filter_input:
            filter_wav_text(args.data_dir, args.train_set)
            filter_wav_text(args.data_dir, args.valid_set)

        if args.dataset_type == "small" and batch_type != "unsorted":
            calc_shape(args, args.train_set)
            calc_shape(args, args.valid_set)

        if args.dataset_type == "large":
            generate_data_list(args, args.data_dir, args.train_set)
            generate_data_list(args, args.data_dir, args.valid_set)

    if distributed:
        dist.barrier()
