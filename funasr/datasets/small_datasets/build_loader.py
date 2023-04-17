import torch
from funasr.datasets.small_datasets.dataset import ESPnetDataset
from funasr.datasets.small_datasets.preprocessor import build_preprocess

def build_dataloader(args, train=False):
    preprocess_fn = build_preprocess(args, train=train)
    dest_sample_rate = args.frontend_conf["fs"] if (args.frontend_conf is not None and "fs" in args.frontend_conf) else 16000
    dataset = ESPnetDataset(
        iter_options.data_path_and_name_and_type,
        float_dtype=args.train_dtype,
        preprocess=preprocess_fn,
        max_cache_size=args.max_cache_size,
        max_cache_fd=args.max_cache_fd,
        dest_sample_rate=dest_sample_rate,
    )
