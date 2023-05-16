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

import argparse
import logging
from optparse import Option
import sys
import json
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Dict

import numpy as np
import torch
from typeguard import check_argument_types

from funasr.fileio.datadir_writer import DatadirWriter
from funasr.datasets.preprocessor import LMPreprocessor
from funasr.tasks.asr import ASRTaskAligner as ASRTask
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none
from funasr.models.frontend.wav_frontend import WavFrontend
from funasr.text.token_id_converter import TokenIDConverter
from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard
from funasr.bin.tp_infer import Speech2Timestamp

def inference_tp(
    batch_size: int,
    ngpu: int,
    log_level: Union[int, str],
    # data_path_and_name_and_type,
    timestamp_infer_config: Optional[str],
    timestamp_model_file: Optional[str],
    timestamp_cmvn_file: Optional[str] = None,
    # raw_inputs: Union[np.ndarray, torch.Tensor] = None,
    key_file: Optional[str] = None,
    allow_variable_data_keys: bool = False,
    output_dir: Optional[str] = None,
    dtype: str = "float32",
    seed: int = 0,
    num_workers: int = 1,
    split_with_space: bool = True,
    seg_dict_file: Optional[str] = None,
    **kwargs,
):
    assert check_argument_types()
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)
    
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    
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
    
    # 2. Build speech2vadsegment
    speechtext2timestamp_kwargs = dict(
        timestamp_infer_config=timestamp_infer_config,
        timestamp_model_file=timestamp_model_file,
        timestamp_cmvn_file=timestamp_cmvn_file,
        device=device,
        dtype=dtype,
    )
    logging.info("speechtext2timestamp_kwargs: {}".format(speechtext2timestamp_kwargs))
    speechtext2timestamp = Speech2Timestamp(**speechtext2timestamp_kwargs)
    
    preprocessor = LMPreprocessor(
        train=False,
        token_type=speechtext2timestamp.tp_train_args.token_type,
        token_list=speechtext2timestamp.tp_train_args.token_list,
        bpemodel=None,
        text_cleaner=None,
        g2p_type=None,
        text_name="text",
        non_linguistic_symbols=speechtext2timestamp.tp_train_args.non_linguistic_symbols,
        split_with_space=split_with_space,
        seg_dict_file=seg_dict_file,
    )
    
    if output_dir is not None:
        writer = DatadirWriter(output_dir)
        tp_writer = writer[f"timestamp_prediction"]
        # ibest_writer["token_list"][""] = " ".join(speech2text.asr_train_args.token_list)
    else:
        tp_writer = None
    
    def _forward(
        data_path_and_name_and_type,
        raw_inputs: Union[np.ndarray, torch.Tensor] = None,
        output_dir_v2: Optional[str] = None,
        fs: dict = None,
        param_dict: dict = None,
        **kwargs
    ):
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        writer = None
        if output_path is not None:
            writer = DatadirWriter(output_path)
            tp_writer = writer[f"timestamp_prediction"]
        else:
            tp_writer = None
        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        
        loader = ASRTask.build_streaming_iterator(
            data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
            preprocess_fn=preprocessor,
            collate_fn=ASRTask.build_collate_fn(speechtext2timestamp.tp_train_args, False),
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )
        
        tp_result_list = []
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            
            logging.info("timestamp predicting, utt_id: {}".format(keys))
            _batch = {'speech': batch['speech'],
                      'speech_lengths': batch['speech_lengths'],
                      'text_lengths': batch['text_lengths']}
            us_alphas, us_cif_peak = speechtext2timestamp(**_batch)
            
            for batch_id in range(_bs):
                key = keys[batch_id]
                token = speechtext2timestamp.converter.ids2tokens(batch['text'][batch_id])
                ts_str, ts_list = ts_prediction_lfr6_standard(us_alphas[batch_id], us_cif_peak[batch_id], token,
                                                              force_time_shift=-3.0)
                logging.warning(ts_str)
                item = {'key': key, 'value': ts_str, 'timestamp': ts_list}
                if tp_writer is not None:
                    tp_writer["tp_sync"][key + '#'] = ts_str
                    tp_writer["tp_time"][key + '#'] = str(ts_list)
                tp_result_list.append(item)
        return tp_result_list
    
    return _forward




def inference_launch(mode, **kwargs):
    if mode == "tp_norm":
        return inference_tp(**kwargs)
    else:
        logging.info("Unknown decoding mode: {}".format(mode))
        return None

def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Timestamp Prediction Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--njob",
        type=int,
        default=1,
        help="The number of jobs for each gpu",
    )
    parser.add_argument(
        "--gpuid_list",
        type=str,
        default="",
        help="The visible gpus",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
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

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--timestamp_infer_config",
        type=str,
        help="VAD infer configuration",
    )
    group.add_argument(
        "--timestamp_model_file",
        type=str,
        help="VAD model parameter file",
    )
    group.add_argument(
        "--timestamp_cmvn_file",
        type=str,
        help="Global CMVN file",
    )

    group = parser.add_argument_group("The inference configuration related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    parser.add_argument(
        "--mode",
        type=str,
        default="tp_norm",
        help="The decoding mode",
    )
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

    inference_pipeline = inference_launch(**kwargs)
    return inference_pipeline(kwargs["data_path_and_name_and_type"])



if __name__ == "__main__":
    main()
