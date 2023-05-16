from funasr.datasets.large_datasets.build_dataloader import LargeDataLoader
from funasr.datasets.small_datasets.sequence_iter_factory import SequenceIterFactory


def build_dataloader(args):
    if args.dataset_type == "small":
        train_iter_factory = SequenceIterFactory(args, mode="train")
        valid_iter_factory = SequenceIterFactory(args, mode="valid")
    elif args.dataset_type == "large":
        train_iter_factory = LargeDataLoader(args, mode="train")
        valid_iter_factory = LargeDataLoader(args, mode="valid")
    else:
        raise ValueError(f"Not supported dataset_type={args.dataset_type}")

    return train_iter_factory, valid_iter_factory
