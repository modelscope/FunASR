# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Abstract task module."""
import argparse
import functools
import logging
import os
import sys
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from distutils.version import LooseVersion
from io import BytesIO
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.nn
import torch.optim
import yaml
from torch.utils.data import DataLoader
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr import __version__
from funasr.datasets.dataset import AbsDataset
from funasr.datasets.dataset import DATA_TYPES
from funasr.datasets.dataset import ESPnetDataset
from funasr.datasets.iterable_dataset import IterableESPnetDataset
from funasr.iterators.abs_iter_factory import AbsIterFactory
from funasr.iterators.chunk_iter_factory import ChunkIterFactory
from funasr.iterators.multiple_iter_factory import MultipleIterFactory
from funasr.iterators.sequence_iter_factory import SequenceIterFactory
from funasr.main_funcs.collect_stats import collect_stats
from funasr.optimizers.sgd import SGD
from funasr.optimizers.fairseq_adam import FairseqAdam
from funasr.samplers.build_batch_sampler import BATCH_TYPES
from funasr.samplers.build_batch_sampler import build_batch_sampler
from funasr.samplers.unsorted_batch_sampler import UnsortedBatchSampler
from funasr.schedulers.noam_lr import NoamLR
from funasr.schedulers.warmup_lr import WarmupLR
from funasr.schedulers.tri_stage_scheduler import TriStageLR
from funasr.torch_utils.load_pretrained_model import load_pretrained_model
from funasr.torch_utils.model_summary import model_summary
from funasr.torch_utils.pytorch_version import pytorch_cudnn_version
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.train.abs_espnet_model import AbsESPnetModel
from funasr.train.class_choices import ClassChoices
from funasr.train.distributed_utils import DistributedOption
from funasr.train.trainer import Trainer
from funasr.utils import config_argparse
from funasr.utils.build_dataclass import build_dataclass
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.get_default_kwargs import get_default_kwargs
from funasr.utils.nested_dict_action import NestedDictAction
from funasr.utils.types import humanfriendly_parse_size_or_none
from funasr.utils.types import int_or_none
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_int
from funasr.utils.types import str_or_none
from funasr.utils.wav_utils import calc_shape, generate_data_list, filter_wav_text
from funasr.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump

try:
    import wandb
except Exception:
    wandb = None

if LooseVersion(torch.__version__) >= LooseVersion("1.5.0"):
    pass
else:
    pass

optim_classes = dict(
    adam=torch.optim.Adam,
    fairseq_adam=FairseqAdam,
    adamw=torch.optim.AdamW,
    sgd=SGD,
    adadelta=torch.optim.Adadelta,
    adagrad=torch.optim.Adagrad,
    adamax=torch.optim.Adamax,
    asgd=torch.optim.ASGD,
    lbfgs=torch.optim.LBFGS,
    rmsprop=torch.optim.RMSprop,
    rprop=torch.optim.Rprop,
)
if LooseVersion(torch.__version__) >= LooseVersion("1.10.0"):
    # From 1.10.0, RAdam is officially supported
    optim_classes.update(
        radam=torch.optim.RAdam,
    )
try:
    import torch_optimizer

    optim_classes.update(
        accagd=torch_optimizer.AccSGD,
        adabound=torch_optimizer.AdaBound,
        adamod=torch_optimizer.AdaMod,
        diffgrad=torch_optimizer.DiffGrad,
        lamb=torch_optimizer.Lamb,
        novograd=torch_optimizer.NovoGrad,
        pid=torch_optimizer.PID,
        # torch_optimizer<=0.0.1a10 doesn't support
        # qhadam=torch_optimizer.QHAdam,
        qhm=torch_optimizer.QHM,
        sgdw=torch_optimizer.SGDW,
        yogi=torch_optimizer.Yogi,
    )
    if LooseVersion(torch_optimizer.__version__) < LooseVersion("0.2.0"):
        # From 0.2.0, RAdam is dropped
        optim_classes.update(
            radam=torch_optimizer.RAdam,
        )
    del torch_optimizer
except ImportError:
    pass
try:
    import apex

    optim_classes.update(
        fusedadam=apex.optimizers.FusedAdam,
        fusedlamb=apex.optimizers.FusedLAMB,
        fusednovograd=apex.optimizers.FusedNovoGrad,
        fusedsgd=apex.optimizers.FusedSGD,
    )
    del apex
except ImportError:
    pass
try:
    import fairscale
except ImportError:
    fairscale = None

scheduler_classes = dict(
    ReduceLROnPlateau=torch.optim.lr_scheduler.ReduceLROnPlateau,
    lambdalr=torch.optim.lr_scheduler.LambdaLR,
    steplr=torch.optim.lr_scheduler.StepLR,
    multisteplr=torch.optim.lr_scheduler.MultiStepLR,
    exponentiallr=torch.optim.lr_scheduler.ExponentialLR,
    CosineAnnealingLR=torch.optim.lr_scheduler.CosineAnnealingLR,
    noamlr=NoamLR,
    warmuplr=WarmupLR,
    tri_stage=TriStageLR,
    cycliclr=torch.optim.lr_scheduler.CyclicLR,
    onecyclelr=torch.optim.lr_scheduler.OneCycleLR,
    CosineAnnealingWarmRestarts=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
)
# To lower keys
optim_classes = {k.lower(): v for k, v in optim_classes.items()}
scheduler_classes = {k.lower(): v for k, v in scheduler_classes.items()}


@dataclass
class IteratorOptions:
    preprocess_fn: callable
    collate_fn: callable
    data_path_and_name_and_type: list
    shape_files: list
    batch_size: int
    batch_bins: int
    batch_type: str
    max_cache_size: float
    max_cache_fd: int
    distributed: bool
    num_batches: Optional[int]
    num_iters_per_epoch: Optional[int]
    train: bool


class AbsTask(ABC):
    # Use @staticmethod, or @classmethod,
    # instead of instance method to avoid God classes

    # If you need more than one optimizers, change this value in inheritance
    num_optimizers: int = 1
    trainer = Trainer
    class_choices_list: List[ClassChoices] = []
    finetune_args: None

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @abstractmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        pass

    @classmethod
    @abstractmethod
    def build_collate_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Callable[[Sequence[Dict[str, np.ndarray]]], Dict[str, torch.Tensor]]:
        """Return "collate_fn", which is a callable object and given to DataLoader.

        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(collate_fn=cls.build_collate_fn(args, train=True), ...)

        In many cases, you can use our common collate_fn.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_preprocess_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def required_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Define the required names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as following,

        >>> from funasr.train.abs_espnet_model import AbsESPnetModel
        >>> class Model(AbsESPnetModel):
        ...     def forward(self, input, output, opt=None):  pass

        then "required_data_names" should be as

        >>> required_data_names = ('input', 'output')
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def optional_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Define the optional names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as follows,

        >>> from funasr.train.abs_espnet_model import AbsESPnetModel
        >>> class Model(AbsESPnetModel):
        ...     def forward(self, input, output, opt=None):  pass

        then "optional_data_names" should be as

        >>> optional_data_names = ('opt',)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_model(cls, args: argparse.Namespace) -> AbsESPnetModel:
        raise NotImplementedError

    @classmethod
    def get_parser(cls) -> config_argparse.ArgumentParser:
        assert check_argument_types()

        class ArgumentDefaultsRawTextHelpFormatter(
            argparse.RawTextHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter,
        ):
            pass

        parser = config_argparse.ArgumentParser(
            description="base parser",
            formatter_class=ArgumentDefaultsRawTextHelpFormatter,
        )

        # NOTE(kamo): Use '_' instead of '-' to avoid confusion.
        #  I think '-' looks really confusing if it's written in yaml.

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        #  to provide --print_config mode. Instead of it, do as
        # parser.set_defaults(required=["output_dir"])

        group = parser.add_argument_group("Common configuration")

        group.add_argument(
            "--print_config",
            action="store_true",
            help="Print the config file and exit",
        )
        group.add_argument(
            "--log_level",
            type=lambda x: x.upper(),
            default="INFO",
            choices=("ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
            help="The verbose level of logging",
        )
        group.add_argument(
            "--dry_run",
            type=str2bool,
            default=False,
            help="Perform process without training",
        )
        group.add_argument(
            "--iterator_type",
            type=str,
            choices=["sequence", "chunk", "task", "none"],
            default="sequence",
            help="Specify iterator type",
        )

        group.add_argument("--output_dir", type=str_or_none, default=None)
        group.add_argument(
            "--ngpu",
            type=int,
            default=0,
            help="The number of gpus. 0 indicates CPU mode",
        )
        group.add_argument("--seed", type=int, default=0, help="Random seed")
        group.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="The number of workers used for DataLoader",
        )
        group.add_argument(
            "--num_att_plot",
            type=int,
            default=3,
            help="The number images to plot the outputs from attention. "
                 "This option makes sense only when attention-based model. "
                 "We can also disable the attention plot by setting it 0",
        )

        group = parser.add_argument_group("distributed training related")
        group.add_argument(
            "--dist_backend",
            default="nccl",
            type=str,
            help="distributed backend",
        )
        group.add_argument(
            "--dist_init_method",
            type=str,
            default="env://",
            help='if init_method="env://", env values of "MASTER_PORT", "MASTER_ADDR", '
                 '"WORLD_SIZE", and "RANK" are referred.',
        )
        group.add_argument(
            "--dist_world_size",
            default=None,
            type=int_or_none,
            help="number of nodes for distributed training",
        )
        group.add_argument(
            "--dist_rank",
            type=int_or_none,
            default=None,
            help="node rank for distributed training",
        )
        group.add_argument(
            # Not starting with "dist_" for compatibility to launch.py
            "--local_rank",
            type=int_or_none,
            default=None,
            help="local rank for distributed training. This option is used if "
                 "--multiprocessing_distributed=false",
        )
        group.add_argument(
            "--dist_master_addr",
            default=None,
            type=str_or_none,
            help="The master address for distributed training. "
                 "This value is used when dist_init_method == 'env://'",
        )
        group.add_argument(
            "--dist_master_port",
            default=None,
            type=int_or_none,
            help="The master port for distributed training"
                 "This value is used when dist_init_method == 'env://'",
        )
        group.add_argument(
            "--dist_launcher",
            default=None,
            type=str_or_none,
            choices=["slurm", "mpi", None],
            help="The launcher type for distributed training",
        )
        group.add_argument(
            "--multiprocessing_distributed",
            default=False,
            type=str2bool,
            help="Use multi-processing distributed training to launch "
                 "N processes per node, which has N GPUs. This is the "
                 "fastest way to use PyTorch for either single node or "
                 "multi node data parallel training",
        )
        group.add_argument(
            "--unused_parameters",
            type=str2bool,
            default=False,
            help="Whether to use the find_unused_parameters in "
                 "torch.nn.parallel.DistributedDataParallel ",
        )
        group.add_argument(
            "--sharded_ddp",
            default=False,
            type=str2bool,
            help="Enable sharded training provided by fairscale",
        )

        group = parser.add_argument_group("cudnn mode related")
        group.add_argument(
            "--cudnn_enabled",
            type=str2bool,
            default=torch.backends.cudnn.enabled,
            help="Enable CUDNN",
        )
        group.add_argument(
            "--cudnn_benchmark",
            type=str2bool,
            default=torch.backends.cudnn.benchmark,
            help="Enable cudnn-benchmark mode",
        )
        group.add_argument(
            "--cudnn_deterministic",
            type=str2bool,
            default=True,
            help="Enable cudnn-deterministic mode",
        )

        group = parser.add_argument_group("collect stats mode related")
        group.add_argument(
            "--collect_stats",
            type=str2bool,
            default=False,
            help='Perform on "collect stats" mode',
        )
        group.add_argument(
            "--write_collected_feats",
            type=str2bool,
            default=False,
            help='Write the output features from the model when "collect stats" mode',
        )

        group = parser.add_argument_group("Trainer related")
        group.add_argument(
            "--max_epoch",
            type=int,
            default=40,
            help="The maximum number epoch to train",
        )
        group.add_argument(
            "--max_update",
            type=int,
            default=sys.maxsize,
            help="The maximum number update step to train",
        )
        group.add_argument(
            "--patience",
            type=int_or_none,
            default=None,
            help="Number of epochs to wait without improvement "
                 "before stopping the training",
        )
        group.add_argument(
            "--val_scheduler_criterion",
            type=str,
            nargs=2,
            default=("valid", "loss"),
            help="The criterion used for the value given to the lr scheduler. "
                 'Give a pair referring the phase, "train" or "valid",'
                 'and the criterion name. The mode specifying "min" or "max" can '
                 "be changed by --scheduler_conf",
        )
        group.add_argument(
            "--early_stopping_criterion",
            type=str,
            nargs=3,
            default=("valid", "loss", "min"),
            help="The criterion used for judging of early stopping. "
                 'Give a pair referring the phase, "train" or "valid",'
                 'the criterion name and the mode, "min" or "max", e.g. "acc,max".',
        )
        group.add_argument(
            "--best_model_criterion",
            type=str2triple_str,
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
        group.add_argument(
            "--keep_nbest_models",
            type=int,
            nargs="+",
            default=[10],
            help="Remove previous snapshots excluding the n-best scored epochs",
        )
        group.add_argument(
            "--nbest_averaging_interval",
            type=int,
            default=0,
            help="The epoch interval to apply model averaging and save nbest models",
        )
        group.add_argument(
            "--grad_clip",
            type=float,
            default=5.0,
            help="Gradient norm threshold to clip",
        )
        group.add_argument(
            "--grad_clip_type",
            type=float,
            default=2.0,
            help="The type of the used p-norm for gradient clip. Can be inf",
        )
        group.add_argument(
            "--grad_noise",
            type=str2bool,
            default=False,
            help="The flag to switch to use noise injection to "
                 "gradients during training",
        )
        group.add_argument(
            "--accum_grad",
            type=int,
            default=1,
            help="The number of gradient accumulation",
        )
        group.add_argument(
            "--no_forward_run",
            type=str2bool,
            default=False,
            help="Just only iterating data loading without "
                 "model forwarding and training",
        )
        group.add_argument(
            "--resume",
            type=str2bool,
            default=False,
            help="Enable resuming if checkpoint is existing",
        )
        group.add_argument(
            "--train_dtype",
            default="float32",
            choices=["float16", "float32", "float64"],
            help="Data type for training.",
        )
        group.add_argument(
            "--use_amp",
            type=str2bool,
            default=False,
            help="Enable Automatic Mixed Precision. This feature requires pytorch>=1.6",
        )
        group.add_argument(
            "--log_interval",
            type=int_or_none,
            default=None,
            help="Show the logs every the number iterations in each epochs at the "
                 "training phase. If None is given, it is decided according the number "
                 "of training samples automatically .",
        )
        group.add_argument(
            "--use_tensorboard",
            type=str2bool,
            default=True,
            help="Enable tensorboard logging",
        )
        group.add_argument(
            "--use_wandb",
            type=str2bool,
            default=False,
            help="Enable wandb logging",
        )
        group.add_argument(
            "--wandb_project",
            type=str,
            default=None,
            help="Specify wandb project",
        )
        group.add_argument(
            "--wandb_id",
            type=str,
            default=None,
            help="Specify wandb id",
        )
        group.add_argument(
            "--wandb_entity",
            type=str,
            default=None,
            help="Specify wandb entity",
        )
        group.add_argument(
            "--wandb_name",
            type=str,
            default=None,
            help="Specify wandb run name",
        )
        group.add_argument(
            "--wandb_model_log_interval",
            type=int,
            default=-1,
            help="Set the model log period",
        )
        group.add_argument(
            "--detect_anomaly",
            type=str2bool,
            default=False,
            help="Set torch.autograd.set_detect_anomaly",
        )

        group = parser.add_argument_group("Pretraining model related")
        group.add_argument("--pretrain_path", help="This option is obsoleted")
        group.add_argument(
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
        group.add_argument(
            "--ignore_init_mismatch",
            type=str2bool,
            default=False,
            help="Ignore size mismatch when loading pre-trained model",
        )
        group.add_argument(
            "--freeze_param",
            type=str,
            default=[],
            nargs="*",
            help="Freeze parameters",
        )

        group = parser.add_argument_group("BatchSampler related")
        group.add_argument(
            "--num_iters_per_epoch",
            type=int_or_none,
            default=None,
            help="Restrict the number of iterations for training per epoch",
        )
        group.add_argument(
            "--batch_size",
            type=int,
            default=20,
            help="The mini-batch size used for training. Used if batch_type='unsorted',"
                 " 'sorted', or 'folded'.",
        )
        group.add_argument(
            "--valid_batch_size",
            type=int_or_none,
            default=None,
            help="If not given, the value of --batch_size is used",
        )
        group.add_argument(
            "--batch_bins",
            type=int,
            default=1000000,
            help="The number of batch bins. Used if batch_type='length' or 'numel'",
        )
        group.add_argument(
            "--valid_batch_bins",
            type=int_or_none,
            default=None,
            help="If not given, the value of --batch_bins is used",
        )

        group.add_argument("--train_shape_file", type=str, action="append", default=[])
        group.add_argument("--valid_shape_file", type=str, action="append", default=[])

        group = parser.add_argument_group("Sequence iterator related")
        _batch_type_help = ""
        for key, value in BATCH_TYPES.items():
            _batch_type_help += f'"{key}":\n{value}\n'
        group.add_argument(
            "--batch_type",
            type=str,
            default="length",
            choices=list(BATCH_TYPES),
            help=_batch_type_help,
        )
        group.add_argument(
            "--valid_batch_type",
            type=str_or_none,
            default=None,
            choices=list(BATCH_TYPES) + [None],
            help="If not given, the value of --batch_type is used",
        )
        group.add_argument(
            "--speech_length_min",
            type=int,
            default=-1,
            help="speech length min",
        )
        group.add_argument(
            "--speech_length_max",
            type=int,
            default=-1,
            help="speech length max",
        )
        group.add_argument("--fold_length", type=int, action="append", default=[])
        group.add_argument(
            "--sort_in_batch",
            type=str,
            default="descending",
            choices=["descending", "ascending"],
            help="Sort the samples in each mini-batches by the sample "
                 'lengths. To enable this, "shape_file" must have the length information.',
        )
        group.add_argument(
            "--sort_batch",
            type=str,
            default="descending",
            choices=["descending", "ascending"],
            help="Sort mini-batches by the sample lengths",
        )
        group.add_argument(
            "--multiple_iterator",
            type=str2bool,
            default=False,
            help="Use multiple iterator mode",
        )

        group = parser.add_argument_group("Chunk iterator related")
        group.add_argument(
            "--chunk_length",
            type=str_or_int,
            default=500,
            help="Specify chunk length. e.g. '300', '300,400,500', or '300-400'."
                 "If multiple numbers separated by command are given, "
                 "one of them is selected randomly for each samples. "
                 "If two numbers are given with '-', it indicates the range of the choices. "
                 "Note that if the sequence length is shorter than the all chunk_lengths, "
                 "the sample is discarded. ",
        )
        group.add_argument(
            "--chunk_shift_ratio",
            type=float,
            default=0.5,
            help="Specify the shift width of chunks. If it's less than 1, "
                 "allows the overlapping and if bigger than 1, there are some gaps "
                 "between each chunk.",
        )
        group.add_argument(
            "--num_cache_chunks",
            type=int,
            default=1024,
            help="Shuffle in the specified number of chunks and generate mini-batches "
                 "More larger this value, more randomness can be obtained.",
        )

        group = parser.add_argument_group("Dataset related")
        _data_path_and_name_and_type_help = (
            "Give three words splitted by comma. It's used for the training data. "
            "e.g. '--train_data_path_and_name_and_type some/path/a.scp,foo,sound'. "
            "The first value, some/path/a.scp, indicates the file path, "
            "and the second, foo, is the key name used for the mini-batch data, "
            "and the last, sound, decides the file type. "
            "This option is repeatable, so you can input any number of features "
            "for your task. Supported file types are as follows:\n\n"
        )
        for key, dic in DATA_TYPES.items():
            _data_path_and_name_and_type_help += f'"{key}":\n{dic["help"]}\n\n'

        # for large dataset
        group.add_argument(
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
        group.add_argument(
            "--train_data_file",
            type=str,
            default=None,
            help="train_list for large dataset",
        )
        group.add_argument(
            "--valid_data_file",
            type=str,
            default=None,
            help="valid_list for large dataset",
        )

        group.add_argument(
            "--train_data_path_and_name_and_type",
            type=str2triple_str,
            action="append",
            default=[],
            help=_data_path_and_name_and_type_help,
        )
        group.add_argument(
            "--valid_data_path_and_name_and_type",
            type=str2triple_str,
            action="append",
            default=[],
        )
        group.add_argument(
            "--allow_variable_data_keys",
            type=str2bool,
            default=False,
            help="Allow the arbitrary keys for mini-batch with ignoring "
                 "the task requirements",
        )
        group.add_argument(
            "--max_cache_size",
            type=humanfriendly.parse_size,
            default=0.0,
            help="The maximum cache size for data loader. e.g. 10MB, 20GB.",
        )
        group.add_argument(
            "--max_cache_fd",
            type=int,
            default=32,
            help="The maximum number of file descriptors to be kept "
                 "as opened for ark files. "
                 "This feature is only valid when data type is 'kaldi_ark'.",
        )
        group.add_argument(
            "--valid_max_cache_size",
            type=humanfriendly_parse_size_or_none,
            default=None,
            help="The maximum cache size for validation data loader. e.g. 10MB, 20GB. "
                 "If None, the 5 percent size of --max_cache_size",
        )

        group = parser.add_argument_group("Optimizer related")
        for i in range(1, cls.num_optimizers + 1):
            suf = "" if i == 1 else str(i)
            group.add_argument(
                f"--optim{suf}",
                type=lambda x: x.lower(),
                default="adadelta",
                choices=list(optim_classes),
                help="The optimizer type",
            )
            group.add_argument(
                f"--optim{suf}_conf",
                action=NestedDictAction,
                default=dict(),
                help="The keyword arguments for optimizer",
            )
            group.add_argument(
                f"--scheduler{suf}",
                type=lambda x: str_or_none(x.lower()),
                default=None,
                choices=list(scheduler_classes) + [None],
                help="The lr scheduler type",
            )
            group.add_argument(
                f"--scheduler{suf}_conf",
                action=NestedDictAction,
                default=dict(),
                help="The keyword arguments for lr scheduler",
            )

        # for training on PAI
        group = parser.add_argument_group("PAI training related")
        group.add_argument(
            "--use_pai",
            type=str2bool,
            default=False,
            help="flag to indicate whether training on PAI",
        )
        group.add_argument(
            "--simple_ddp",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--num_worker_count",
            type=int,
            default=1,
            help="The number of machines on PAI.",
        )
        group.add_argument(
            "--access_key_id",
            type=str,
            default=None,
            help="The username for oss.",
        )
        group.add_argument(
            "--access_key_secret",
            type=str,
            default=None,
            help="The password for oss.",
        )
        group.add_argument(
            "--endpoint",
            type=str,
            default=None,
            help="The endpoint for oss.",
        )
        group.add_argument(
            "--bucket_name",
            type=str,
            default=None,
            help="The bucket name for oss.",
        )
        group.add_argument(
            "--oss_bucket",
            default=None,
            help="oss bucket.",
        )

        cls.trainer.add_arguments(parser)
        cls.add_task_arguments(parser)

        assert check_return_type(parser)
        return parser

    @classmethod
    def build_optimizers(
            cls,
            args: argparse.Namespace,
            model: torch.nn.Module,
    ) -> List[torch.optim.Optimizer]:
        if cls.num_optimizers != 1:
            raise RuntimeError(
                "build_optimizers() must be overridden if num_optimizers != 1"
            )

        optim_class = optim_classes.get(args.optim)
        if optim_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        if args.sharded_ddp:
            if fairscale is None:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim = fairscale.optim.oss.OSS(
                params=model.parameters(), optim=optim_class, **args.optim_conf
            )
        else:
            optim = optim_class(model.parameters(), **args.optim_conf)

        optimizers = [optim]
        return optimizers

    @classmethod
    def exclude_opts(cls) -> Tuple[str, ...]:
        """The options not to be shown by --print_config"""
        return "required", "print_config", "config", "ngpu"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return the configuration as dict.

        This method is used by print_config()
        """

        def get_class_type(name: str, classes: dict):
            _cls = classes.get(name)
            if _cls is None:
                raise ValueError(f"must be one of {list(classes)}: {name}")
            return _cls

        # This method is used only for --print_config
        assert check_argument_types()
        parser = cls.get_parser()
        args, _ = parser.parse_known_args()
        config = vars(args)
        # Excludes the options not to be shown
        for k in AbsTask.exclude_opts():
            config.pop(k)

        for i in range(1, cls.num_optimizers + 1):
            suf = "" if i == 1 else str(i)
            name = config[f"optim{suf}"]
            optim_class = get_class_type(name, optim_classes)
            conf = get_default_kwargs(optim_class)
            # Overwrite the default by the arguments,
            conf.update(config[f"optim{suf}_conf"])
            # and set it again
            config[f"optim{suf}_conf"] = conf

            name = config[f"scheduler{suf}"]
            if name is not None:
                scheduler_class = get_class_type(name, scheduler_classes)
                conf = get_default_kwargs(scheduler_class)
                # Overwrite the default by the arguments,
                conf.update(config[f"scheduler{suf}_conf"])
                # and set it again
                config[f"scheduler{suf}_conf"] = conf

        for class_choices in cls.class_choices_list:
            if getattr(args, class_choices.name) is not None:
                class_obj = class_choices.get_class(getattr(args, class_choices.name))
                conf = get_default_kwargs(class_obj)
                name = class_choices.name
                # Overwrite the default by the arguments,
                conf.update(config[f"{name}_conf"])
                # and set it again
                config[f"{name}_conf"] = conf
        return config

    @classmethod
    def check_required_command_args(cls, args: argparse.Namespace):
        assert check_argument_types()
        if hasattr(args, "required"):
            for k in vars(args):
                if "-" in k:
                    raise RuntimeError(f'Use "_" instead of "-": parser.get_parser("{k}")')

            required = ", ".join(
                f"--{a}" for a in args.required if getattr(args, a) is None
            )

            if len(required) != 0:
                parser = cls.get_parser()
                parser.print_help(file=sys.stderr)
                p = Path(sys.argv[0]).name
                print(file=sys.stderr)
                print(
                    f"{p}: error: the following arguments are required: " f"{required}",
                    file=sys.stderr,
                )
                sys.exit(2)

    @classmethod
    def check_task_requirements(
            cls,
            dataset: Union[AbsDataset, IterableESPnetDataset],
            allow_variable_data_keys: bool,
            train: bool,
            inference: bool = False,
    ) -> None:
        """Check if the dataset satisfy the requirement of current Task"""
        assert check_argument_types()
        mes = (
            f"If you intend to use an additional input, modify "
            f'"{cls.__name__}.required_data_names()" or '
            f'"{cls.__name__}.optional_data_names()". '
            f"Otherwise you need to set --allow_variable_data_keys true "
        )

        for k in cls.required_data_names(train, inference):
            if not dataset.has_name(k):
                raise RuntimeError(
                    f'"{cls.required_data_names(train, inference)}" are required for'
                    f' {cls.__name__}. but "{dataset.names()}" are input.\n{mes}'
                )
        if not allow_variable_data_keys:
            task_keys = cls.required_data_names(
                train, inference
            ) + cls.optional_data_names(train, inference)
            for k in dataset.names():
                if k not in task_keys:
                    raise RuntimeError(
                        f"The data-name must be one of {task_keys} "
                        f'for {cls.__name__}: "{k}" is not allowed.\n{mes}'
                    )

    @classmethod
    def print_config(cls, file=sys.stdout) -> None:
        assert check_argument_types()
        # Shows the config: e.g. python train.py asr --print_config
        config = cls.get_default_config()
        file.write(yaml_no_alias_safe_dump(config, indent=4, sort_keys=False))

    @classmethod
    def main(cls, args: argparse.Namespace = None, cmd: Sequence[str] = None):
        assert check_argument_types()
        print(get_commandline_args(), file=sys.stderr)
        if args is None:
            parser = cls.get_parser()
            args = parser.parse_args(cmd)
        args.version = __version__
        if args.pretrain_path is not None:
            raise RuntimeError("--pretrain_path is deprecated. Use --init_param")
        if args.print_config:
            cls.print_config()
            sys.exit(0)
        cls.check_required_command_args(args)

        if not args.distributed or not args.multiprocessing_distributed:
            cls.main_worker(args)
        else:
            assert args.ngpu > 1
            cls.main_worker(args)

    @classmethod
    def run(cls):
        assert hasattr(cls, "finetune_args")
        args = cls.finetune_args
        args.train_shape_file = None
        if args.distributed:
            args.simple_ddp = True
        else:
            args.simple_ddp = False
            args.ngpu = 1
        args.use_pai = False
        args.batch_type = "length"
        args.oss_bucket = None
        args.input_size = None
        cls.main_worker(args)

    @classmethod
    def main_worker(cls, args: argparse.Namespace):
        assert check_argument_types()

        # 0. Init distributed process
        distributed_option = build_dataclass(DistributedOption, args)
        # Setting distributed_option.dist_rank, etc.
        if args.use_pai:
            distributed_option.init_options_pai()
        elif not args.simple_ddp:
            distributed_option.init_options()

        # Invoking torch.distributed.init_process_group
        if args.use_pai:
            distributed_option.init_torch_distributed_pai(args)
        elif not args.simple_ddp:
            distributed_option.init_torch_distributed(args)
        elif args.distributed and args.simple_ddp:
            distributed_option.init_torch_distributed_pai(args)
            args.ngpu = dist.get_world_size()
            if args.dataset_type == "small":
                if args.batch_size is not None:
                    args.batch_size = args.batch_size * args.ngpu
                if args.batch_bins is not None:
                    args.batch_bins = args.batch_bins * args.ngpu

        # filter samples if wav.scp and text are mismatch
        if (args.train_shape_file is None and args.dataset_type == "small") or args.train_data_file is None and args.dataset_type == "large":
            if not args.simple_ddp or distributed_option.dist_rank == 0:
                filter_wav_text(args.data_dir, args.train_set)
                filter_wav_text(args.data_dir, args.dev_set)
            if args.simple_ddp:
                dist.barrier()

        if args.train_shape_file is None and args.dataset_type == "small":
            if not args.simple_ddp or distributed_option.dist_rank == 0:
                calc_shape(args.data_dir, args.train_set, args.frontend_conf, args.speech_length_min, args.speech_length_max)
                calc_shape(args.data_dir, args.dev_set, args.frontend_conf, args.speech_length_min, args.speech_length_max)
            if args.simple_ddp:
                dist.barrier()
            args.train_shape_file = [os.path.join(args.data_dir, args.train_set, "speech_shape")]
            args.valid_shape_file = [os.path.join(args.data_dir, args.dev_set, "speech_shape")]

        if args.train_data_file is None and args.dataset_type == "large":
            if not args.simple_ddp or distributed_option.dist_rank == 0:
                generate_data_list(args.data_dir, args.train_set)
                generate_data_list(args.data_dir, args.dev_set)
            if args.simple_ddp:
                dist.barrier()
            args.train_data_file = os.path.join(args.data_dir, args.train_set, "data.list")
            args.valid_data_file = os.path.join(args.data_dir, args.dev_set, "data.list")

        # NOTE(kamo): Don't use logging before invoking logging.basicConfig()
        if not distributed_option.distributed or distributed_option.dist_rank == 0:
            if not distributed_option.distributed:
                _rank = ""
            else:
                _rank = (
                    f":{distributed_option.dist_rank}/"
                    f"{distributed_option.dist_world_size}"
                )

            # NOTE(kamo):
            # logging.basicConfig() is invoked in main_worker() instead of main()
            # because it can be invoked only once in a process.
            # FIXME(kamo): Should we use logging.getLogger()?
            logging.basicConfig(
                level=args.log_level,
                format=f"[{os.uname()[1].split('.')[0]}]"
                       f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )
        else:
            # Suppress logging if RANK != 0
            logging.basicConfig(
                level="ERROR",
                format=f"[{os.uname()[1].split('.')[0]}]"
                       f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )
        logging.info("world size: {}, rank: {}, local_rank: {}".format(distributed_option.dist_world_size,
                                                                       distributed_option.dist_rank,
                                                                       distributed_option.local_rank))

        # 1. Set random-seed
        set_all_random_seed(args.seed)
        torch.backends.cudnn.enabled = args.cudnn_enabled
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cudnn.deterministic = args.cudnn_deterministic
        if args.detect_anomaly:
            logging.info("Invoking torch.autograd.set_detect_anomaly(True)")
            torch.autograd.set_detect_anomaly(args.detect_anomaly)

        # 2. Build model
        model = cls.build_model(args=args)
        if not isinstance(model, AbsESPnetModel):
            raise RuntimeError(
                f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
            )
        model = model.to(
            dtype=getattr(torch, args.train_dtype),
            device="cuda" if args.ngpu > 0 else "cpu",
        )
        for t in args.freeze_param:
            for k, p in model.named_parameters():
                if k.startswith(t + ".") or k == t:
                    logging.info(f"Setting {k}.requires_grad = False")
                    p.requires_grad = False

        # 3. Build optimizer
        optimizers = cls.build_optimizers(args, model=model)

        # 4. Build schedulers
        schedulers = []
        for i, optim in enumerate(optimizers, 1):
            suf = "" if i == 1 else str(i)
            name = getattr(args, f"scheduler{suf}")
            conf = getattr(args, f"scheduler{suf}_conf")
            if name is not None:
                cls_ = scheduler_classes.get(name)
                if cls_ is None:
                    raise ValueError(
                        f"must be one of {list(scheduler_classes)}: {name}"
                    )
                scheduler = cls_(optim, **conf)
            else:
                scheduler = None

            schedulers.append(scheduler)

        logging.info(pytorch_cudnn_version())
        logging.info(model_summary(model))
        for i, (o, s) in enumerate(zip(optimizers, schedulers), 1):
            suf = "" if i == 1 else str(i)
            logging.info(f"Optimizer{suf}:\n{o}")
            logging.info(f"Scheduler{suf}: {s}")

        # 5. Dump "args" to config.yaml
        # NOTE(kamo): "args" should be saved after object-buildings are done
        #  because they are allowed to modify "args".
        output_dir = Path(args.output_dir)
        if not distributed_option.distributed or distributed_option.dist_rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
                logging.info(
                    f'Saving the configuration in {output_dir / "config.yaml"}'
                )
                if args.use_pai:
                    buffer = BytesIO()
                    torch.save({"config": vars(args)}, buffer)
                    args.oss_bucket.put_object(os.path.join(args.output_dir, "config.dict"), buffer.getvalue())
                else:
                    yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

        if args.dry_run:
            pass
        elif args.collect_stats:
            # Perform on collect_stats mode. This mode has two roles
            # - Derive the length and dimension of all input data
            # - Accumulate feats, square values, and the length for whitening

            if args.valid_batch_size is None:
                args.valid_batch_size = args.batch_size

            if len(args.train_shape_file) != 0:
                train_key_file = args.train_shape_file[0]
            else:
                train_key_file = None
            if len(args.valid_shape_file) != 0:
                valid_key_file = args.valid_shape_file[0]
            else:
                valid_key_file = None

            collect_stats(
                model=model,
                train_iter=cls.build_streaming_iterator(
                    data_path_and_name_and_type=args.train_data_path_and_name_and_type,
                    key_file=train_key_file,
                    batch_size=args.batch_size,
                    dtype=args.train_dtype,
                    num_workers=args.num_workers,
                    allow_variable_data_keys=args.allow_variable_data_keys,
                    ngpu=args.ngpu,
                    preprocess_fn=cls.build_preprocess_fn(args, train=False),
                    collate_fn=cls.build_collate_fn(args, train=False),
                ),
                valid_iter=cls.build_streaming_iterator(
                    data_path_and_name_and_type=args.valid_data_path_and_name_and_type,
                    key_file=valid_key_file,
                    batch_size=args.valid_batch_size,
                    dtype=args.train_dtype,
                    num_workers=args.num_workers,
                    allow_variable_data_keys=args.allow_variable_data_keys,
                    ngpu=args.ngpu,
                    preprocess_fn=cls.build_preprocess_fn(args, train=False),
                    collate_fn=cls.build_collate_fn(args, train=False),
                ),
                output_dir=output_dir,
                ngpu=args.ngpu,
                log_interval=args.log_interval,
                write_collected_feats=args.write_collected_feats,
            )
        else:
            logging.info("Training args: {}".format(args))
            # 6. Loads pre-trained model
            for p in args.init_param:
                logging.info(f"Loading pretrained params from {p}")
                load_pretrained_model(
                    model=model,
                    init_param=p,
                    ignore_init_mismatch=args.ignore_init_mismatch,
                    # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
                    #   in PyTorch<=1.4
                    map_location=f"cuda:{torch.cuda.current_device()}"
                    if args.ngpu > 0
                    else "cpu",
                    oss_bucket=args.oss_bucket,
                )

            # 7. Build iterator factories
            if args.dataset_type == "large":
                from funasr.datasets.large_datasets.build_dataloader import ArkDataLoader
                train_iter_factory = ArkDataLoader(args.train_data_file, args.token_list, args.dataset_conf,
                                                   frontend_conf=args.frontend_conf if hasattr(args, "frontend_conf") else None,
                                                   seg_dict_file=args.seg_dict_file if hasattr(args,
                                                                                               "seg_dict_file") else None,
                                                   punc_dict_file=args.punc_list if hasattr(args, "punc_list") else None,
                                                   mode="train")
                valid_iter_factory = ArkDataLoader(args.valid_data_file, args.token_list, args.dataset_conf, 
                                                   frontend_conf=args.frontend_conf if hasattr(args, "frontend_conf") else None,
                                                   seg_dict_file=args.seg_dict_file if hasattr(args,
                                                                                               "seg_dict_file") else None,
                                                   punc_dict_file=args.punc_list if hasattr(args, "punc_list") else None,
                                                   mode="eval")
            elif args.dataset_type == "small":
                train_iter_factory = cls.build_iter_factory(
                    args=args,
                    distributed_option=distributed_option,
                    mode="train",
                )
                valid_iter_factory = cls.build_iter_factory(
                    args=args,
                    distributed_option=distributed_option,
                    mode="valid",
                )
            else:
                raise ValueError(f"Not supported dataset_type={args.dataset_type}")

            if args.scheduler == "tri_stage":
                for scheduler in schedulers:
                    scheduler.init_tri_stage_scheudler(max_update=args.max_update)

            # 8. Start training
            if args.use_wandb:
                if wandb is None:
                    raise RuntimeError("Please install wandb")

                try:
                    wandb.login()
                except wandb.errors.UsageError:
                    logging.info("wandb not configured! run `wandb login` to enable")
                    args.use_wandb = False

            if args.use_wandb:
                if (
                        not distributed_option.distributed
                        or distributed_option.dist_rank == 0
                ):
                    if args.wandb_project is None:
                        project = "FunASR_" + cls.__name__
                    else:
                        project = args.wandb_project

                    if args.wandb_name is None:
                        name = str(Path(".").resolve()).replace("/", "_")
                    else:
                        name = args.wandb_name

                    wandb.init(
                        entity=args.wandb_entity,
                        project=project,
                        name=name,
                        dir=output_dir,
                        id=args.wandb_id,
                        resume="allow",
                    )
                    wandb.config.update(args)
                else:
                    # wandb also supports grouping for distributed training,
                    # but we only logs aggregated data,
                    # so it's enough to perform on rank0 node.
                    args.use_wandb = False

            # Don't give args to trainer.run() directly!!!
            # Instead of it, define "Options" object and build here.
            trainer_options = cls.trainer.build_options(args)
            cls.trainer.run(
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                train_iter_factory=train_iter_factory,
                valid_iter_factory=valid_iter_factory,
                trainer_options=trainer_options,
                distributed_option=distributed_option,
            )

            if args.use_wandb and wandb.run:
                wandb.finish()

    @classmethod
    def build_iter_options(
            cls,
            args: argparse.Namespace,
            distributed_option: DistributedOption,
            mode: str,
    ):
        if mode == "train":
            preprocess_fn = cls.build_preprocess_fn(args, train=True)
            collate_fn = cls.build_collate_fn(args, train=True)
            data_path_and_name_and_type = args.train_data_path_and_name_and_type
            shape_files = args.train_shape_file
            batch_size = args.batch_size
            batch_bins = args.batch_bins
            batch_type = args.batch_type
            max_cache_size = args.max_cache_size
            max_cache_fd = args.max_cache_fd
            distributed = distributed_option.distributed
            num_batches = None
            num_iters_per_epoch = args.num_iters_per_epoch
            train = True

        elif mode == "valid":
            preprocess_fn = cls.build_preprocess_fn(args, train=False)
            collate_fn = cls.build_collate_fn(args, train=False)
            data_path_and_name_and_type = args.valid_data_path_and_name_and_type
            shape_files = args.valid_shape_file

            if args.valid_batch_type is None:
                batch_type = args.batch_type
            else:
                batch_type = args.valid_batch_type
            if args.valid_batch_size is None:
                batch_size = args.batch_size
            else:
                batch_size = args.valid_batch_size
            if args.valid_batch_bins is None:
                batch_bins = args.batch_bins
            else:
                batch_bins = args.valid_batch_bins
            if args.valid_max_cache_size is None:
                # Cache 5% of maximum size for validation loader
                max_cache_size = 0.05 * args.max_cache_size
            else:
                max_cache_size = args.valid_max_cache_size
            max_cache_fd = args.max_cache_fd
            distributed = distributed_option.distributed
            num_batches = None
            num_iters_per_epoch = None
            train = False
        else:
            raise NotImplementedError(f"mode={mode}")

        return IteratorOptions(
            preprocess_fn=preprocess_fn,
            collate_fn=collate_fn,
            data_path_and_name_and_type=data_path_and_name_and_type,
            shape_files=shape_files,
            batch_type=batch_type,
            batch_size=batch_size,
            batch_bins=batch_bins,
            num_batches=num_batches,
            max_cache_size=max_cache_size,
            max_cache_fd=max_cache_fd,
            distributed=distributed,
            num_iters_per_epoch=num_iters_per_epoch,
            train=train,
        )

    @classmethod
    def build_iter_factory(
            cls,
            args: argparse.Namespace,
            distributed_option: DistributedOption,
            mode: str,
            kwargs: dict = None,
    ) -> AbsIterFactory:
        """Build a factory object of mini-batch iterator.

        This object is invoked at every epochs to build the iterator for each epoch
        as following:

        >>> iter_factory = cls.build_iter_factory(...)
        >>> for epoch in range(1, max_epoch):
        ...     for keys, batch in iter_fatory.build_iter(epoch):
        ...         model(**batch)

        The mini-batches for each epochs are fully controlled by this class.
        Note that the random seed used for shuffling is decided as "seed + epoch" and
        the generated mini-batches can be reproduces when resuming.

        Note that the definition of "epoch" doesn't always indicate
        to run out of the whole training corpus.
        "--num_iters_per_epoch" option restricts the number of iterations for each epoch
        and the rest of samples for the originally epoch are left for the next epoch.
        e.g. If The number of mini-batches equals to 4, the following two are same:

        - 1 epoch without "--num_iters_per_epoch"
        - 4 epoch with "--num_iters_per_epoch" == 4

        """
        assert check_argument_types()
        iter_options = cls.build_iter_options(args, distributed_option, mode)

        # Overwrite iter_options if any kwargs is given
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(iter_options, k, v)

        if args.iterator_type == "sequence":
            return cls.build_sequence_iter_factory(
                args=args,
                iter_options=iter_options,
                mode=mode,
            )
        elif args.iterator_type == "chunk":
            return cls.build_chunk_iter_factory(
                args=args,
                iter_options=iter_options,
                mode=mode,
            )
        elif args.iterator_type == "task":
            return cls.build_task_iter_factory(
                args=args,
                iter_options=iter_options,
                mode=mode,
            )
        else:
            raise RuntimeError(f"Not supported: iterator_type={args.iterator_type}")

    @classmethod
    def build_sequence_iter_factory(
            cls, args: argparse.Namespace, iter_options: IteratorOptions, mode: str
    ) -> AbsIterFactory:
        assert check_argument_types()

        dataset = ESPnetDataset(
            iter_options.data_path_and_name_and_type,
            float_dtype=args.train_dtype,
            preprocess=iter_options.preprocess_fn,
            max_cache_size=iter_options.max_cache_size,
            max_cache_fd=iter_options.max_cache_fd,
            dest_sample_rate=args.frontend_conf["fs"] if args.frontend_conf else 16000,
        )
        cls.check_task_requirements(
            dataset, args.allow_variable_data_keys, train=iter_options.train
        )

        if Path(
                Path(iter_options.data_path_and_name_and_type[0][0]).parent, "utt2category"
        ).exists():
            utt2category_file = str(
                Path(
                    Path(iter_options.data_path_and_name_and_type[0][0]).parent,
                    "utt2category",
                )
            )
        else:
            utt2category_file = None
        batch_sampler = build_batch_sampler(
            type=iter_options.batch_type,
            shape_files=iter_options.shape_files,
            fold_lengths=args.fold_length,
            batch_size=iter_options.batch_size,
            batch_bins=iter_options.batch_bins,
            sort_in_batch=args.sort_in_batch,
            sort_batch=args.sort_batch,
            drop_last=False,
            min_batch_size=torch.distributed.get_world_size()
            if iter_options.distributed
            else 1,
            utt2category_file=utt2category_file,
        )

        batches = list(batch_sampler)
        if iter_options.num_batches is not None:
            batches = batches[: iter_options.num_batches]

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

        if iter_options.distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            for batch in batches:
                if len(batch) < world_size:
                    raise RuntimeError(
                        f"The batch-size must be equal or more than world_size: "
                        f"{len(batch)} < {world_size}"
                    )
            batches = [batch[rank::world_size] for batch in batches]

        return SequenceIterFactory(
            dataset=dataset,
            batches=batches,
            seed=args.seed,
            num_iters_per_epoch=iter_options.num_iters_per_epoch,
            shuffle=iter_options.train,
            num_workers=args.num_workers,
            collate_fn=iter_options.collate_fn,
            pin_memory=args.ngpu > 0,
        )

    @classmethod
    def build_chunk_iter_factory(
            cls,
            args: argparse.Namespace,
            iter_options: IteratorOptions,
            mode: str,
    ) -> AbsIterFactory:
        assert check_argument_types()

        dataset = ESPnetDataset(
            iter_options.data_path_and_name_and_type,
            float_dtype=args.train_dtype,
            preprocess=iter_options.preprocess_fn,
            max_cache_size=iter_options.max_cache_size,
            max_cache_fd=iter_options.max_cache_fd,
        )
        cls.check_task_requirements(
            dataset, args.allow_variable_data_keys, train=iter_options.train
        )

        if len(iter_options.shape_files) == 0:
            key_file = iter_options.data_path_and_name_and_type[0][0]
        else:
            key_file = iter_options.shape_files[0]

        batch_sampler = UnsortedBatchSampler(batch_size=1, key_file=key_file)
        batches = list(batch_sampler)
        if iter_options.num_batches is not None:
            batches = batches[: iter_options.num_batches]
        logging.info(f"[{mode}] dataset:\n{dataset}")

        if iter_options.distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            if len(batches) < world_size:
                raise RuntimeError("Number of samples is smaller than world_size")
            if iter_options.batch_size < world_size:
                raise RuntimeError("batch_size must be equal or more than world_size")

            if rank < iter_options.batch_size % world_size:
                batch_size = iter_options.batch_size // world_size + 1
            else:
                batch_size = iter_options.batch_size // world_size
            num_cache_chunks = args.num_cache_chunks // world_size
            # NOTE(kamo): Split whole corpus by sample numbers without considering
            #   each of the lengths, therefore the number of iteration counts are not
            #   always equal to each other and the iterations are limitted
            #   by the fewest iterations.
            #   i.e. the samples over the counts are discarded.
            batches = batches[rank::world_size]
        else:
            batch_size = iter_options.batch_size
            num_cache_chunks = args.num_cache_chunks

        return ChunkIterFactory(
            dataset=dataset,
            batches=batches,
            seed=args.seed,
            batch_size=batch_size,
            # For chunk iterator,
            # --num_iters_per_epoch doesn't indicate the number of iterations,
            # but indicates the number of samples.
            num_samples_per_epoch=iter_options.num_iters_per_epoch,
            shuffle=iter_options.train,
            num_workers=args.num_workers,
            collate_fn=iter_options.collate_fn,
            pin_memory=args.ngpu > 0,
            chunk_length=args.chunk_length,
            chunk_shift_ratio=args.chunk_shift_ratio,
            num_cache_chunks=num_cache_chunks,
        )

    # NOTE(kamo): Not abstract class
    @classmethod
    def build_task_iter_factory(
            cls,
            args: argparse.Namespace,
            iter_options: IteratorOptions,
            mode: str,
    ) -> AbsIterFactory:
        """Build task specific iterator factory

        Example:

            >>> class YourTask(AbsTask):
            ... @classmethod
            ... def add_task_arguments(cls, parser: argparse.ArgumentParser):
            ...     parser.set_defaults(iterator_type="task")
            ...
            ... @classmethod
            ... def build_task_iter_factory(
            ...     cls,
            ...     args: argparse.Namespace,
            ...     iter_options: IteratorOptions,
            ...     mode: str,
            ... ):
            ...     return FooIterFactory(...)
            ...
            ... @classmethod
            ... def build_iter_options(
            ....    args: argparse.Namespace,
            ...     distributed_option: DistributedOption,
            ...     mode: str
            ... ):
            ...     # if you need to customize options object
        """
        raise NotImplementedError

    @classmethod
    def build_multiple_iter_factory(
            cls, args: argparse.Namespace, distributed_option: DistributedOption, mode: str
    ):
        assert check_argument_types()
        iter_options = cls.build_iter_options(args, distributed_option, mode)
        assert len(iter_options.data_path_and_name_and_type) > 0, len(
            iter_options.data_path_and_name_and_type
        )

        # 1. Sanity check
        num_splits = None
        for path in [
                        path for path, _, _ in iter_options.data_path_and_name_and_type
                    ] + list(iter_options.shape_files):
            if not Path(path).is_dir():
                raise RuntimeError(f"{path} is not a directory")
            p = Path(path) / "num_splits"
            if not p.exists():
                raise FileNotFoundError(f"{p} is not found")
            with p.open() as f:
                _num_splits = int(f.read())
                if num_splits is not None and num_splits != _num_splits:
                    raise RuntimeError(
                        f"Number of splits are mismathed: "
                        f"{iter_options.data_path_and_name_and_type[0][0]} and {path}"
                    )
                num_splits = _num_splits

            for i in range(num_splits):
                p = Path(path) / f"split.{i}"
                if not p.exists():
                    raise FileNotFoundError(f"{p} is not found")

        # 2. Create functions to build an iter factory for each splits
        data_path_and_name_and_type_list = [
            [
                (str(Path(p) / f"split.{i}"), n, t)
                for p, n, t in iter_options.data_path_and_name_and_type
            ]
            for i in range(num_splits)
        ]
        shape_files_list = [
            [str(Path(s) / f"split.{i}") for s in iter_options.shape_files]
            for i in range(num_splits)
        ]
        num_iters_per_epoch_list = [
            (iter_options.num_iters_per_epoch + i) // num_splits
            if iter_options.num_iters_per_epoch is not None
            else None
            for i in range(num_splits)
        ]
        max_cache_size = iter_options.max_cache_size / num_splits

        # Note that iter-factories are built for each epoch at runtime lazily.
        build_funcs = [
            functools.partial(
                cls.build_iter_factory,
                args,
                distributed_option,
                mode,
                kwargs=dict(
                    data_path_and_name_and_type=_data_path_and_name_and_type,
                    shape_files=_shape_files,
                    num_iters_per_epoch=_num_iters_per_epoch,
                    max_cache_size=max_cache_size,
                ),
            )
            for (
                _data_path_and_name_and_type,
                _shape_files,
                _num_iters_per_epoch,
            ) in zip(
                data_path_and_name_and_type_list,
                shape_files_list,
                num_iters_per_epoch_list,
            )
        ]

        # 3. Build MultipleIterFactory
        return MultipleIterFactory(
            build_funcs=build_funcs, shuffle=iter_options.train, seed=args.seed
        )

    @classmethod
    def build_streaming_iterator(
            cls,
            data_path_and_name_and_type,
            preprocess_fn,
            collate_fn,
            key_file: str = None,
            batch_size: int = 1,
            fs: dict = None,
            mc: bool = False,
            dtype: str = np.float32,
            num_workers: int = 1,
            allow_variable_data_keys: bool = False,
            ngpu: int = 0,
            inference: bool = False,
    ) -> DataLoader:
        """Build DataLoader using iterable dataset"""
        assert check_argument_types()
        # For backward compatibility for pytorch DataLoader
        if collate_fn is not None:
            kwargs = dict(collate_fn=collate_fn)
        else:
            kwargs = {}

        dataset = IterableESPnetDataset(
            data_path_and_name_and_type,
            float_dtype=dtype,
            fs=fs,
            mc=mc,
            preprocess=preprocess_fn,
            key_file=key_file,
        )
        if dataset.apply_utt2category:
            kwargs.update(batch_size=1)
        else:
            kwargs.update(batch_size=batch_size)

        cls.check_task_requirements(
            dataset, allow_variable_data_keys, train=False, inference=inference
        )

        return DataLoader(
            dataset=dataset,
            pin_memory=ngpu > 0,
            num_workers=num_workers,
            **kwargs,
        )

    # ~~~~~~~~~ The methods below are mainly used for inference ~~~~~~~~~
    @classmethod
    def build_model_from_file(
            cls,
            config_file: Union[Path, str] = None,
            model_file: Union[Path, str] = None,
            cmvn_file: Union[Path, str] = None,
            device: str = "cpu",
    ) -> Tuple[AbsESPnetModel, argparse.Namespace]:
        """Build model from the files.

        This method is used for inference or fine-tuning.

        Args:
            config_file: The yaml file saved when training.
            model_file: The model file saved when training.
            device: Device type, "cpu", "cuda", or "cuda:N".

        """
        assert check_argument_types()
        if config_file is None:
            assert model_file is not None, (
                "The argument 'model_file' must be provided "
                "if the argument 'config_file' is not specified."
            )
            config_file = Path(model_file).parent / "config.yaml"
        else:
            config_file = Path(config_file)

        with config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        if cmvn_file is not None:
            args["cmvn_file"] = cmvn_file
        args = argparse.Namespace(**args)
        model = cls.build_model(args)
        if not isinstance(model, AbsESPnetModel):
            raise RuntimeError(
                f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
            )
        model.to(device)
        if model_file is not None:
            if device == "cuda":
                # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
                #   in PyTorch<=1.4
                device = f"cuda:{torch.cuda.current_device()}"
            model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        return model, args
