import logging
import os

import numpy as np
import torch

from funasr.datasets.small_datasets.dataset import ESPnetDataset
from funasr.datasets.small_datasets.preprocessor import build_preprocess
from funasr.datasets.small_datasets.length_batch_sampler import LengthBatchSampler


def build_dataloader(args, mode="train"):
    preprocess_fn = build_preprocess(args, train=mode == "train")
    dest_sample_rate = args.frontend_conf["fs"] if (
            args.frontend_conf is not None and "fs" in args.frontend_conf) else 16000
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

    dataset_conf = args.dataset_conf
    batch_sampler = LengthBatchSampler(
        batch_bins=dataset_conf["batch_size"],
        shape_files=shape_files,
        sort_in_batch=dataset_conf["sort_in_batch"] if hasattr(dataset_conf, "sort_in_batch") else "descending",
        sort_batch=dataset_conf["sort_batch"] if hasattr(dataset_conf, "sort_batch") else "ascending",
        drop_last=False,
        padding=True,
    )

    batches = list(batch_sampler)
    bs_list = [len(batch) for batch in batches]
    logging.info(f"[{mode}] dataset:\n{dataset}")
    logging.info(f"[{mode}] Batch sampler: {batch_sampler}")
    logging.info(
        f"[{mode}] mini-batch sizes summary: N-batch={len(bs_list)}, "
        f"mean={np.mean(bs_list):.1f}, min={np.min(bs_list)}, max={np.max(bs_list)}"
    )

    if args.scheduler == "tri_stage" and mode == "train":
        args.max_update = len(bs_list) * args.max_epoch
        logging.info("Max update: {}".format(args.max_update))

    if args.distributed:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        for batch in batches:
            if len(batch) < world_size:
                raise RuntimeError(
                    f"The batch-size must be equal or more than world_size: "
                    f"{len(batch)} < {world_size}"
                )
        batches = [batch[rank::world_size] for batch in batches]
