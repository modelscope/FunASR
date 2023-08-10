#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


import argparse
import logging
import os
import sys
from typing import Optional
from typing import Union

import numpy as np
import torch
import soundfile as sf
from funasr.build_utils.build_streaming_iterator import build_streaming_iterator
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.types import str2triple_str
from funasr.bin.ss_infer import SpeechSeparator


def inference_ss(
        batch_size: int,
        ngpu: int,
        log_level: Union[int, str],
        ss_infer_config: Optional[str],
        ss_model_file: Optional[str],
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        num_workers: int = 1,
        num_spks: int = 2,
        sample_rate: int = 8000,
        param_dict: dict = None,
        **kwargs,
):
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size = 1

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech separator
    speech_separator_kwargs = dict(
        ss_infer_config=ss_infer_config,
        ss_model_file=ss_model_file,
        device=device,
        dtype=dtype,
    )
    logging.info("speech_separator_kwargs: {}".format(speech_separator_kwargs))
    speech_separator = SpeechSeparator(**speech_separator_kwargs)

    def _forward(
            data_path_and_name_and_type,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            output_dir_v2: Optional[str] = None,
            fs: dict = None,
            param_dict: dict = None
    ):
        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = build_streaming_iterator(
            task_name="ss",
            preprocess_args=None,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            fs=fs,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # 4 .Start for-loop
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if not os.path.exists(output_path):
            cmd = 'mkdir -p ' + output_path 
            os.system(cmd)       
 
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            # do speech separation
            logging.info('decoding: {}'.format(keys[0]))
            ss_results = speech_separator(**batch)
            
            for spk in range(num_spks):
                sf.write(os.path.join(output_path, keys[0] + '_s' + str(spk+1)+'.wav'), ss_results[spk], sample_rate)
        torch.cuda.empty_cache()
        return ss_results

    return _forward


def inference_launch(mode, **kwargs):
    if mode == "mossformer":
        return inference_ss(**kwargs)
    else:
        logging.info("Unknown decoding mode: {}".format(mode))
        return None


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Speech Separator Decoding",
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

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=1,
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
        default="2",
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

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--ss_infer_config",
        type=str,
        help="SS infer configuration",
    )
    group.add_argument(
        "--ss_model_file",
        type=str,
        help="SS model parameter file",
    )
    group.add_argument(
        "--ss_train_config",
        type=str,
        help="SS training configuration",
    )

    group = parser.add_argument_group("The inference configuration related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    parser.add_argument(
        '--num-spks', dest='num_spks', type=int, default=2)

    parser.add_argument(
        '--one-time-decode-length', dest='one_time_decode_length', type=int,
        default=60, help='the max length (second) for one-time decoding')

    parser.add_argument(
        '--decode-window', dest='decode_window', type=int,
        default=1, help='segmental decoding window length (second)')

    parser.add_argument(
        '--sample-rate', dest='sample_rate', type=int, default='8000')
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    parser.add_argument(
        "--mode",
        type=str,
        default="mossformer",
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

