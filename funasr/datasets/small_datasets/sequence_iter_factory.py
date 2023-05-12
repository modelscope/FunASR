import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from funasr.datasets.small_datasets.collate_fn import CommonCollateFn
from funasr.datasets.small_datasets.dataset import ESPnetDataset
from funasr.datasets.small_datasets.length_batch_sampler import LengthBatchSampler
from funasr.datasets.small_datasets.preprocessor import build_preprocess
from funasr.iterators.abs_iter_factory import AbsIterFactory
from funasr.samplers.abs_sampler import AbsSampler


class RawSampler(AbsSampler):
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def generate(self, seed):
        return list(self.batches)


class SequenceIterFactory(AbsIterFactory):
    """Build iterator for each epoch, modified from ESPnet

    """

    def __init__(self, args, mode="train"):

        # preprocess
        preprocess_fn = build_preprocess(args, train=mode == "train")

        # collate
        if args.task_name in ["punc", "lm"]:
            collate_fn = CommonCollateFn(int_pad_value=0)
        else:
            collate_fn = CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

        # dataset
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
            speed_perturb=args.speed_perturb,
        )

        # sampler
        dataset_conf = args.dataset_conf
        batch_sampler = LengthBatchSampler(
            batch_bins=dataset_conf["batch_conf"]["batch_size"] * args.ngpu,
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

        if args.distributed and mode=="train":
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            for batch in batches:
                if len(batch) < world_size:
                    raise RuntimeError(
                        f"The batch-size must be equal or more than world_size: "
                        f"{len(batch)} < {world_size}"
                    )
            batches = [batch[rank::world_size] for batch in batches]

        if not isinstance(batches, AbsSampler):
            self.sampler = RawSampler(batches)
        else:
            self.sampler = batches

        self.dataset = dataset
        self.num_iters_per_epoch = None
        self.shuffle = mode == "train"
        self.seed = args.seed
        self.num_workers = args.dataset_conf.get("num_workers", 8)
        self.collate_fn = collate_fn
        self.pin_memory = args.ngpu > 0

    def build_iter(self, epoch: int, shuffle: bool = None) -> DataLoader:
        if shuffle is None:
            shuffle = self.shuffle

        if self.num_iters_per_epoch is not None:
            N = len(self.sampler)
            # If corpus size is larger than the num_per_epoch
            if self.num_iters_per_epoch < N:
                N = len(self.sampler)
                real_epoch, offset = divmod(self.num_iters_per_epoch * epoch, N)

                if offset >= self.num_iters_per_epoch:
                    current_batches = self.sampler.generate(real_epoch + self.seed)
                    if shuffle:
                        np.random.RandomState(real_epoch + self.seed).shuffle(
                            current_batches
                        )
                    batches = current_batches[
                              offset - self.num_iters_per_epoch: offset
                              ]
                else:
                    prev_batches = self.sampler.generate(real_epoch - 1 + self.seed)
                    current_batches = self.sampler.generate(real_epoch + self.seed)
                    if shuffle:
                        np.random.RandomState(real_epoch - 1 + self.seed).shuffle(
                            prev_batches
                        )
                        np.random.RandomState(real_epoch + self.seed).shuffle(
                            current_batches
                        )
                    batches = (
                            prev_batches[offset - self.num_iters_per_epoch:]
                            + current_batches[:offset]
                    )

            # If corpus size is less than the num_per_epoch
            else:
                _epoch, _cursor = divmod(self.num_iters_per_epoch * (epoch - 1), N)
                _remain = self.num_iters_per_epoch
                batches = []
                current_batches = self.sampler.generate(_epoch + self.seed)
                if shuffle:
                    np.random.RandomState(_epoch + self.seed).shuffle(current_batches)
                while _remain > 0:

                    _batches = current_batches[_cursor: _cursor + _remain]
                    batches += _batches
                    if _cursor + _remain >= N:
                        _epoch += 1
                        _cursor = 0
                        current_batches = self.sampler.generate(_epoch + self.seed)
                        if shuffle:
                            np.random.RandomState(_epoch + self.seed).shuffle(
                                current_batches
                            )
                    else:
                        _cursor = _cursor + _remain
                    _remain -= len(_batches)

                assert len(batches) == self.num_iters_per_epoch

        else:
            batches = self.sampler.generate(epoch + self.seed)
            if shuffle:
                np.random.RandomState(epoch + self.seed).shuffle(batches)

        # For backward compatibility for pytorch DataLoader
        if self.collate_fn is not None:
            kwargs = dict(collate_fn=self.collate_fn)
        else:
            kwargs = {}

        return DataLoader(
            dataset=self.dataset,
            batch_sampler=batches,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **kwargs,
        )
