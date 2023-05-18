# -*- encoding: utf-8 -*-
#!/usr/bin/env python3
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

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

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Any
from typing import List

import numpy as np
import torch
from typeguard import check_argument_types

from funasr.datasets.preprocessor import CodeMixTokenizerCommonPreprocessor
from funasr.utils.cli_utils import get_commandline_args
from funasr.tasks.punctuation import PunctuationTask
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.forward_adaptor import ForwardAdaptor
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none
from funasr.datasets.preprocessor import split_to_mini_sentence
from funasr.bin.punc_infer import Text2Punc, Text2PuncVADRealtime

def inference_punc(
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    output_dir: Optional[str] = None,
    param_dict: dict = None,
    **kwargs,
):
    assert check_argument_types()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)
    text2punc = Text2Punc(train_config, model_file, device)

    def _forward(
        data_path_and_name_and_type,
        raw_inputs: Union[List[Any], bytes, str] = None,
        output_dir_v2: Optional[str] = None,
        cache: List[Any] = None,
        param_dict: dict = None,
    ):
        results = []
        split_size = 20

        if raw_inputs != None:
            line = raw_inputs.strip()
            key = "demo"
            if line == "":
                item = {'key': key, 'value': ""}
                results.append(item)
                return results
            result, _ = text2punc(line)
            item = {'key': key, 'value': result}
            results.append(item)
            return results

        for inference_text, _, _ in data_path_and_name_and_type:
            with open(inference_text, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    segs = line.split("\t")
                    if len(segs) != 2:
                        continue
                    key = segs[0]
                    if len(segs[1]) == 0:
                        continue
                    result, _ = text2punc(segs[1])
                    item = {'key': key, 'value': result}
                    results.append(item)
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path != None:
            output_file_name = "infer.out"
            Path(output_path).mkdir(parents=True, exist_ok=True)
            output_file_path = (Path(output_path) / output_file_name).absolute()
            with open(output_file_path, "w", encoding="utf-8") as fout:
                for item_i in results:
                    key_out = item_i["key"]
                    value_out = item_i["value"]
                    fout.write(f"{key_out}\t{value_out}\n")
        return results

    return _forward

def inference_punc_vad_realtime(
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    #cache: list,
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    output_dir: Optional[str] = None,
    param_dict: dict = None,
    **kwargs,
):
    assert check_argument_types()
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)
    text2punc = Text2PuncVADRealtime(train_config, model_file, device)

    def _forward(
        data_path_and_name_and_type,
        raw_inputs: Union[List[Any], bytes, str] = None,
        output_dir_v2: Optional[str] = None,
        cache: List[Any] = None,
        param_dict: dict = None,
    ):
        results = []
        split_size = 10
        cache_in = param_dict["cache"]
        if raw_inputs != None:
            line = raw_inputs.strip()
            key = "demo"
            if line == "":
                item = {'key': key, 'value': ""}
                results.append(item)
                return results
            result, _, cache = text2punc(line, cache_in)
            param_dict["cache"] = cache
            item = {'key': key, 'value': result}
            results.append(item)
            return results

        return results

    return _forward



def inference_launch(mode, **kwargs):
    if mode == "punc":
        return inference_punc(**kwargs)
    if mode == "punc_VadRealtime":
        return inference_punc_vad_realtime(**kwargs)
    else:
        logging.info("Unknown decoding mode: {}".format(mode))
        return None

def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Punctuation inference",
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

    group = parser.add_argument_group("Input data related")
    group.add_argument("--data_path_and_name_and_type", type=str2triple_str, action="append", required=False)
    group.add_argument("--raw_inputs", type=str, required=False)
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--cache", type=list, required=False)
    group.add_argument("--param_dict", type=dict, required=False)
    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--train_config", type=str)
    group.add_argument("--model_file", type=str)
    group.add_argument("--mode", type=str, default="punc")
    return parser


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
    inference_pipeline = inference_launch(**kwargs)
    return inference_pipeline(kwargs["data_path_and_name_and_type"])



if __name__ == "__main__":
    main()
