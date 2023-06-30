#!/usr/bin/env python3
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import argparse
import logging
import os
import sys
from io import BytesIO

import torch

from funasr.build_utils.build_args import build_args
from funasr.build_utils.build_dataloader import build_dataloader
from funasr.build_utils.build_distributed import build_distributed
from funasr.build_utils.build_model import build_model
from funasr.build_utils.build_optimizer import build_optimizer
from funasr.build_utils.build_scheduler import build_scheduler
from funasr.build_utils.build_trainer import build_trainer
from funasr.text.phoneme_tokenizer import g2p_choices
from funasr.torch_utils.load_pretrained_model import load_pretrained_model
from funasr.torch_utils.model_summary import model_summary
from funasr.torch_utils.pytorch_version import pytorch_cudnn_version
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils.nested_dict_action import NestedDictAction
from funasr.utils.prepare_data import prepare_data
from funasr.utils.types import int_or_none
from funasr.utils.types import str2bool
from funasr.utils.types import str_or_none
from funasr.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump


def get_parser():
    parser = argparse.ArgumentParser(
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
        type=int,
        default=1,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--dist_rank",
        type=int,
        default=None,
        help="node rank for distributed training",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="local rank for distributed training",
    )
    parser.add_argument(
        "--dist_master_addr",
        default=None,
        type=str_or_none,
        help="The master address for distributed training. "
             "This value is used when dist_init_method == 'env://'",
    )
    parser.add_argument(
        "--dist_master_port",
        default=None,
        type=int_or_none,
        help="The master port for distributed training"
             "This value is used when dist_init_method == 'env://'",
    )
    parser.add_argument(
        "--dist_launcher",
        default=None,
        type=str_or_none,
        choices=["slurm", "mpi", None],
        help="The launcher type for distributed training",
    )
    parser.add_argument(
        "--multiprocessing_distributed",
        default=True,
        type=str2bool,
        help="Use multi-processing distributed training to launch "
             "N processes per node, which has N GPUs. This is the "
             "fastest way to use PyTorch for either single node or "
             "multi node data parallel training",
    )
    parser.add_argument(
        "--unused_parameters",
        type=str2bool,
        default=False,
        help="Whether to use the find_unused_parameters in "
             "torch.nn.parallel.DistributedDataParallel ",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="local gpu id.",
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
        type=int_or_none,
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
        "--train_dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type for training.",
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
    parser.add_argument(
        "--use_tensorboard",
        type=str2bool,
        default=True,
        help="Enable tensorboard logging",
    )

    # pretrained model related
    parser.add_argument(
        "--init_param",
        type=str,
        action="append",
        default=[],
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
        action="append",
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
        "--dataset_conf",
        action=NestedDictAction,
        default=dict(),
        help=f"The keyword arguments for dataset",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="root path of data",
    )
    parser.add_argument(
        "--train_set",
        type=str,
        default="train",
        help="train dataset",
    )
    parser.add_argument(
        "--valid_set",
        type=str,
        default="validation",
        help="dev dataset",
    )
    parser.add_argument(
        "--data_file_names",
        type=str,
        default="wav.scp,text",
        help="input data files",
    )
    parser.add_argument(
        "--speed_perturb",
        type=float,
        nargs="+",
        default=None,
        help="speed perturb",
    )
    parser.add_argument(
        "--use_preprocessor",
        type=str2bool,
        default=True,
        help="Apply preprocessing to data or not",
    )

    # optimization related
    parser.add_argument(
        "--optim",
        type=lambda x: x.lower(),
        default="adam",
        help="The optimizer type",
    )
    parser.add_argument(
        "--optim_conf",
        action=NestedDictAction,
        default=dict(),
        help="The keyword arguments for optimizer",
    )
    parser.add_argument(
        "--scheduler",
        type=lambda x: str_or_none(x.lower()),
        default=None,
        help="The lr scheduler type",
    )
    parser.add_argument(
        "--scheduler_conf",
        action=NestedDictAction,
        default=dict(),
        help="The keyword arguments for lr scheduler",
    )

    # most task related
    parser.add_argument(
        "--init",
        type=lambda x: str_or_none(x.lower()),
        default=None,
        help="The initialization method",
        choices=[
            "chainer",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            None,
        ],
    )
    parser.add_argument(
        "--token_list",
        type=str_or_none,
        default=None,
        help="A text mapping int-id to token",
    )
    parser.add_argument(
        "--token_type",
        type=str,
        default="bpe",
        choices=["bpe", "char", "word"],
        help="",
    )
    parser.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model file fo sentencepiece",
    )
    parser.add_argument(
        "--cleaner",
        type=str_or_none,
        choices=[None, "tacotron", "jaconv", "vietnamese"],
        default=None,
        help="Apply text cleaning",
    )
    parser.add_argument(
        "--g2p",
        type=str_or_none,
        choices=g2p_choices,
        default=None,
        help="Specify g2p method if --token_type=phn",
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

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args, extra_task_params = parser.parse_known_args()
    if extra_task_params:
        args = build_args(args, parser, extra_task_params)

    # set random seed
    set_all_random_seed(args.seed)
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = args.cudnn_deterministic

    # ddp init
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.distributed = args.ngpu > 1 or args.dist_world_size > 1
    distributed_option = build_distributed(args)

    # for logging
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

    model = build_model(args)
    model = model.to(
        dtype=getattr(torch, args.train_dtype),
        device="cuda" if args.ngpu > 0 else "cpu",
    )
    for t in args.freeze_param:
        for k, p in model.named_parameters():
            if k.startswith(t + ".") or k == t:
                logging.info(f"Setting {k}.requires_grad = False")
                p.requires_grad = False

    optimizers = build_optimizer(args, model=model)
    schedulers = build_scheduler(args, optimizers)

    logging.info("world size: {}, rank: {}, local_rank: {}".format(distributed_option.dist_world_size,
                                                                   distributed_option.dist_rank,
                                                                   distributed_option.local_rank))
    logging.info(pytorch_cudnn_version())
    logging.info("Args: {}".format(args))
    logging.info(model_summary(model))
    logging.info("Optimizer: {}".format(optimizers))
    logging.info("Scheduler: {}".format(schedulers))

    # dump args to config.yaml
    if not distributed_option.distributed or distributed_option.dist_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
            logging.info("Saving the configuration in {}/{}".format(args.output_dir, "config.yaml"))
            if args.use_pai:
                buffer = BytesIO()
                torch.save({"config": vars(args)}, buffer)
                args.oss_bucket.put_object(os.path.join(args.output_dir, "config.dict"), buffer.getvalue())
            else:
                yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

    for p in args.init_param:
        logging.info(f"Loading pretrained params from {p}")
        load_pretrained_model(
            model=model,
            init_param=p,
            ignore_init_mismatch=args.ignore_init_mismatch,
            map_location=f"cuda:{torch.cuda.current_device()}"
            if args.ngpu > 0
            else "cpu",
            oss_bucket=args.oss_bucket,
        )

    # dataloader for training/validation
    train_dataloader, valid_dataloader = build_dataloader(args)

    # Trainer, including model, optimizers, etc.
    trainer = build_trainer(
        args=args,
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        distributed_option=distributed_option
    )

    trainer.run()
