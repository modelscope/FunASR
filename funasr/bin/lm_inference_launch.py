#!/usr/bin/env python3
# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
torch.set_num_threads(1)

import argparse
import logging
import os
import sys
from typing import Union, Dict, Any

from funasr.utils import config_argparse
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none
from funasr.utils.types import float_or_none


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Calc perplexity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpuid_list", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--njob", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    parser.add_argument(
        "--log_base",
        type=float_or_none,
        default=10,
        help="The base of logarithm for Perplexity. "
             "If None, napier's constant is used.",
        required=False
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        action="append",
        required=False
    )
    group.add_argument(
        "--raw_inputs",
        type=str,
        required=False
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group.add_argument("--split_with_space", type=str2bool, default=False)
    group.add_argument("--seg_dict_file", type=str_or_none)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--train_config", type=str)
    group.add_argument("--model_file", type=str)
    group.add_argument("--mode", type=str, default="lm")
    return parser

def inference_launch(mode, **kwargs):
    if mode == "transformer":
        from funasr.bin.lm_inference import inference_modelscope
        return inference_modelscope(**kwargs)
    else:
        logging.info("Unknown decoding mode: {}".format(mode))
        return None


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)

    # set logging messages
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logging.info("Decoding args: {}".format(kwargs))

    # gpu setting
    if args.ngpu > 0:
        jobid = int(args.output_dir.split(".")[-1])
        gpuid = args.gpuid_list.split(",")[(jobid - 1) // args.njob]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

    kwargs.pop("gpuid_list", None)
    kwargs.pop("njob", None)
    results = inference_launch(**kwargs)


if __name__ == "__main__":
    main()

