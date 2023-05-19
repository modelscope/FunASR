import logging
import os

import torch

from funasr.train.distributed_utils import DistributedOption
from funasr.utils.build_dataclass import build_dataclass


def build_distributed(args):
    distributed_option = build_dataclass(DistributedOption, args)
    if args.use_pai:
        distributed_option.init_options_pai()
        distributed_option.init_torch_distributed_pai(args)
    elif not args.simple_ddp:
        distributed_option.init_torch_distributed(args)
    elif args.distributed and args.simple_ddp:
        distributed_option.init_torch_distributed_pai(args)
        args.ngpu = torch.distributed.get_world_size()

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
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
    logging.info("world size: {}, rank: {}, local_rank: {}".format(distributed_option.dist_world_size,
                                                                   distributed_option.dist_rank,
                                                                   distributed_option.local_rank))
    return distributed_option
