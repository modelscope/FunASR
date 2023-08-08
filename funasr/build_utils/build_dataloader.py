from funasr.datasets.large_datasets.build_dataloader import LargeDataLoader
from funasr.datasets.small_datasets.sequence_iter_factory import SequenceIterFactory


def build_dataloader(args):
    if args.dataset_type == "small":
        if args.task_name == "diar" and args.model == "eend_ola":
            from funasr.modules.eend_ola.eend_ola_dataloader import EENDOLADataLoader
            train_iter_factory = EENDOLADataLoader(
                data_file=args.train_data_path_and_name_and_type[0][0],
                batch_size=args.dataset_conf["batch_conf"]["batch_size"],
                num_workers=args.dataset_conf["num_workers"],
                shuffle=True)
            valid_iter_factory = EENDOLADataLoader(
                data_file=args.valid_data_path_and_name_and_type[0][0],
                batch_size=args.dataset_conf["batch_conf"]["batch_size"],
                num_workers=0,
                shuffle=False)
        else:
            train_iter_factory = SequenceIterFactory(args, mode="train")
            valid_iter_factory = SequenceIterFactory(args, mode="valid")
    elif args.dataset_type == "large":
        train_iter_factory = LargeDataLoader(args, mode="train")
        valid_iter_factory = LargeDataLoader(args, mode="valid")
    else:
        raise ValueError(f"Not supported dataset_type={args.dataset_type}")

    return train_iter_factory, valid_iter_factory
