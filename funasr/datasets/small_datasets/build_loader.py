import os

import torch
from funasr.datasets.small_datasets.dataset import ESPnetDataset
from funasr.datasets.small_datasets.preprocessor import build_preprocess
from funasr.samplers.build_batch_sampler import build_batch_sampler

def build_dataloader(args, mode="train"):
    preprocess_fn = build_preprocess(args, train=mode=="train")
    dest_sample_rate = args.frontend_conf["fs"] if (args.frontend_conf is not None and "fs" in args.frontend_conf) else 16000
    if mode == "train":
        data_path_and_name_and_type = args.train_data_path_and_name_and_type
        shape_files = args.train_shape_file
    elif mode == "valid":
        data_path_and_name_and_type = args.valid_data_path_and_name_and_type
        shape_files = args.valid_shape_file
    else:
        raise NotImplementedError(f"mode={mode}")
    dataset = ESPnetDataset(
        data_path_and_name_and_type,
        preprocess=preprocess_fn,
        dest_sample_rate=dest_sample_rate,
    )
    if os.path.exists(os.path.join(data_path_and_name_and_type[0][0].parent, "utt2category")):
        utt2category_file = os.path.join(data_path_and_name_and_type[0][0].parent, "utt2category")
    else:
        utt2category_file = None
    batch_sampler = build_batch_sampler(
        type=args.batch_type,
        shape_files=iter_options.shape_files,
        fold_lengths=args.fold_length,
        batch_size=iter_options.batch_size,
        batch_bins=iter_options.batch_bins,
        sort_in_batch=args.sort_in_batch,
        sort_batch=args.sort_batch,
        drop_last=False,
        min_batch_size=torch.distributed.get_world_size() if args.distributed else 1,
        utt2category_file=utt2category_file,
    )