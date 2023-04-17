from funasr.datasets.large_datasets.build_dataloader import LargeDataLoader


def build_dataloader(args):
    if args.dataset_type == "small":
        pass
    elif args.dataset_type == "large":
        train_iter_factory = LargeDataLoader(args, mode="train")
        valid_iter_factory = LargeDataLoader(args,  mode="valid")
    else:
        raise ValueError(f"Not supported dataset_type={args.dataset_type}")