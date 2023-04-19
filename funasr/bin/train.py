import logging
import os
import sys

import torch

from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.build_dataloader import build_dataloader
from funasr.utils.build_distributed import build_distributed
from funasr.utils.prepare_data import prepare_data
from funasr.utils.build_optimizer import build_optimizer
from funasr.utils.build_scheduler import build_scheduler
from funasr.utils.types import str2bool


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="FunASR Common Training Parser",
    )

    # common configuration
    parser.add_argument("--output_dir", help="model save path")
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--task_name", type=str, default="asr", help="Name for different tasks")

    # ddp related
    parser.add_argument(
        "--dist_backend",
        default="nccl",
        type=str,
        help="distributed backend",
    )
    parser.add_argument(
        "--dist_init_method",
        type=str,
        default="env://",
        help='if init_method="env://", env values of "MASTER_PORT", "MASTER_ADDR", '
             '"WORLD_SIZE", and "RANK" are referred.',
    )
    parser.add_argument(
        "--dist_world_size",
        default=None,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--dist_rank",
        default=None,
        help="node rank for distributed training",
    )
    parser.add_argument(
        "--local_rank",
        default=None,
        help="local rank for distributed training",
    )
    parser.add_argument(
        "--unused_parameters",
        type=str2bool,
        default=False,
        help="Whether to use the find_unused_parameters in "
             "torch.nn.parallel.DistributedDataParallel ",
    )

    # cudnn related
    parser.add_argument(
        "--cudnn_enabled",
        type=str2bool,
        default=torch.backends.cudnn.enabled,
        help="Enable CUDNN",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        type=str2bool,
        default=torch.backends.cudnn.benchmark,
        help="Enable cudnn-benchmark mode",
    )
    parser.add_argument(
        "--cudnn_deterministic",
        type=str2bool,
        default=True,
        help="Enable cudnn-deterministic mode",
    )

    # trainer related
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=40,
        help="The maximum number epoch to train",
    )
    parser.add_argument(
        "--max_update",
        type=int,
        default=sys.maxsize,
        help="The maximum number update step to train",
    )
    parser.add_argument(
        "--batch_interval",
        type=int,
        default=10000,
        help="The batch interval for saving model.",
    )
    parser.add_argument(
        "--patience",
        default=None,
        help="Number of epochs to wait without improvement "
             "before stopping the training",
    )
    parser.add_argument(
        "--val_scheduler_criterion",
        type=str,
        nargs=2,
        default=("valid", "loss"),
        help="The criterion used for the value given to the lr scheduler. "
             'Give a pair referring the phase, "train" or "valid",'
             'and the criterion name. The mode specifying "min" or "max" can '
             "be changed by --scheduler_conf",
    )
    parser.add_argument(
        "--early_stopping_criterion",
        type=str,
        nargs=3,
        default=("valid", "loss", "min"),
        help="The criterion used for judging of early stopping. "
             'Give a pair referring the phase, "train" or "valid",'
             'the criterion name and the mode, "min" or "max", e.g. "acc,max".',
    )
    parser.add_argument(
        "--best_model_criterion",
        nargs="+",
        default=[
            ("train", "loss", "min"),
            ("valid", "loss", "min"),
            ("train", "acc", "max"),
            ("valid", "acc", "max"),
        ],
        help="The criterion used for judging of the best model. "
             'Give a pair referring the phase, "train" or "valid",'
             'the criterion name, and the mode, "min" or "max", e.g. "acc,max".',
    )
    parser.add_argument(
        "--keep_nbest_models",
        type=int,
        nargs="+",
        default=[10],
        help="Remove previous snapshots excluding the n-best scored epochs",
    )
    parser.add_argument(
        "--nbest_averaging_interval",
        type=int,
        default=0,
        help="The epoch interval to apply model averaging and save nbest models",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
        help="Gradient norm threshold to clip",
    )
    parser.add_argument(
        "--grad_clip_type",
        type=float,
        default=2.0,
        help="The type of the used p-norm for gradient clip. Can be inf",
    )
    parser.add_argument(
        "--grad_noise",
        type=str2bool,
        default=False,
        help="The flag to switch to use noise injection to "
             "gradients during training",
    )
    parser.add_argument(
        "--accum_grad",
        type=int,
        default=1,
        help="The number of gradient accumulation",
    )
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=False,
        help="Enable resuming if checkpoint is existing",
    )
    parser.add_argument(
        "--use_amp",
        type=str2bool,
        default=False,
        help="Enable Automatic Mixed Precision. This feature requires pytorch>=1.6",
    )
    parser.add_argument(
        "--log_interval",
        default=None,
        help="Show the logs every the number iterations in each epochs at the "
             "training phase. If None is given, it is decided according the number "
             "of training samples automatically .",
    )

    # pretrained model related
    parser.add_argument(
        "--init_param",
        type=str,
        default=[],
        nargs="*",
        help="Specify the file path used for initialization of parameters. "
             "The format is '<file_path>:<src_key>:<dst_key>:<exclude_keys>', "
             "where file_path is the model file path, "
             "src_key specifies the key of model states to be used in the model file, "
             "dst_key specifies the attribute of the model to be initialized, "
             "and exclude_keys excludes keys of model states for the initialization."
             "e.g.\n"
             "  # Load all parameters"
             "  --init_param some/where/model.pb\n"
             "  # Load only decoder parameters"
             "  --init_param some/where/model.pb:decoder:decoder\n"
             "  # Load only decoder parameters excluding decoder.embed"
             "  --init_param some/where/model.pb:decoder:decoder:decoder.embed\n"
             "  --init_param some/where/model.pb:decoder:decoder:decoder.embed\n",
    )
    parser.add_argument(
        "--ignore_init_mismatch",
        type=str2bool,
        default=False,
        help="Ignore size mismatch when loading pre-trained model",
    )
    parser.add_argument(
        "--freeze_param",
        type=str,
        default=[],
        nargs="*",
        help="Freeze parameters",
    )

    # dataset related
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="small",
        help="whether to use dataloader for large dataset",
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        default=None,
        help="train_list for large dataset",
    )
    parser.add_argument(
        "--valid_data_file",
        type=str,
        default=None,
        help="valid_list for large dataset",
    )
    parser.add_argument(
        "--train_data_path_and_name_and_type",
        action="append",
        default=[],
        help="e.g. '--train_data_path_and_name_and_type some/path/a.scp,foo,sound'. ",
    )
    parser.add_argument(
        "--valid_data_path_and_name_and_type",
        action="append",
        default=[],
    )

    # pai related
    parser.add_argument(
        "--use_pai",
        type=str2bool,
        default=False,
        help="flag to indicate whether training on PAI",
    )
    parser.add_argument(
        "--simple_ddp",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--num_worker_count",
        type=int,
        default=1,
        help="The number of machines on PAI.",
    )
    parser.add_argument(
        "--access_key_id",
        type=str,
        default=None,
        help="The username for oss.",
    )
    parser.add_argument(
        "--access_key_secret",
        type=str,
        default=None,
        help="The password for oss.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="The endpoint for oss.",
    )
    parser.add_argument(
        "--bucket_name",
        type=str,
        default=None,
        help="The bucket name for oss.",
    )
    parser.add_argument(
        "--oss_bucket",
        default=None,
        help="oss bucket.",
    )

    # task related
    parser.add_argument("--task_name", help="for different task")

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # ddp init
    args.distributed = args.dist_world_size > 1
    distributed_option = build_distributed(args)
    if not distributed_option.distributed or distributed_option.dist_rank == 0:
        logging.basicConfig(
            level="INFO",
            format=f"[{os.uname()[1].split('.')[0]}]"
                   f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level="ERROR",
            format=f"[{os.uname()[1].split('.')[0]}]"
                   f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

    # prepare files for dataloader
    prepare_data(args, distributed_option)

    # set random seed
    set_all_random_seed(args.seed)
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = args.cudnn_deterministic

    train_dataloader, valid_dataloader = build_dataloader(args)

    logging.info("world size: {}, rank: {}, local_rank: {}".format(distributed_option.dist_world_size,
                                                                   distributed_option.dist_rank,
                                                                   distributed_option.local_rank))

    model = build_model(args)
    optimizers = build_optimizer(args, model=model)
    schedule = build_scheduler(args)
