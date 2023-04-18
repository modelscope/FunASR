from funasr.datasets.large_datasets.build_dataloader import LargeDataLoader
from funasr.datasets.small_datasets.build_dataloader import build_dataloader

def build_dataloader(args):
    if args.dataset_type == "small":
        train_iter_factory = LargeDataLoader(args, mode="train")
        valid_iter_factory = LargeDataLoader(args, mode="valid")
    elif args.dataset_type == "large":
        train_iter_factory = LargeDataLoader(args, mode="train")
        valid_iter_factory = LargeDataLoader(args,  mode="valid")
    else:
        raise ValueError(f"Not supported dataset_type={args.dataset_type}")