#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import argparse
import logging
import os
import sys
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from kaldiio import WriteHelper
from typeguard import check_argument_types

from funasr.bin.sv_infer import Speech2Xvector
from funasr.build_utils.build_streaming_iterator import build_streaming_iterator
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none


def inference_sv(
        output_dir: Optional[str] = None,
        batch_size: int = 1,
        dtype: str = "float32",
        ngpu: int = 1,
        seed: int = 0,
        num_workers: int = 0,
        log_level: Union[int, str] = "INFO",
        key_file: Optional[str] = None,
        sv_train_config: Optional[str] = "sv.yaml",
        sv_model_file: Optional[str] = "sv.pb",
        model_tag: Optional[str] = None,
        allow_variable_data_keys: bool = True,
        streaming: bool = False,
        embedding_node: str = "resnet1_dense",
        sv_threshold: float = 0.9465,
        param_dict: Optional[dict] = None,
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
    logging.info("param_dict: {}".format(param_dict))

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2xvector
    speech2xvector_kwargs = dict(
        sv_train_config=sv_train_config,
        sv_model_file=sv_model_file,
        device=device,
        dtype=dtype,
        streaming=streaming,
        embedding_node=embedding_node
    )
    logging.info("speech2xvector_kwargs: {}".format(speech2xvector_kwargs))
    speech2xvector = Speech2Xvector.from_pretrained(
        model_tag=model_tag,
        **speech2xvector_kwargs,
    )
    speech2xvector.sv_model.eval()

    def _forward(
            data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            output_dir_v2: Optional[str] = None,
            param_dict: Optional[dict] = None,
    ):
        logging.info("param_dict: {}".format(param_dict))
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]

        # 3. Build data-iterator
        loader = build_streaming_iterator(
            task_name="sv",
            preprocess_args=None,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
            use_collate_fn=False,
        )

        # 7 .Start for-loop
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        embd_writer, ref_embd_writer, score_writer = None, None, None
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            embd_writer = WriteHelper("ark,scp:{}/xvector.ark,{}/xvector.scp".format(output_path, output_path))
        sv_result_list = []
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            embedding, ref_embedding, score = speech2xvector(**batch)
            # Only supporting batch_size==1
            key = keys[0]
            normalized_score = 0.0
            if score is not None:
                score = score.item()
                normalized_score = max(score - sv_threshold, 0.0) / (1.0 - sv_threshold) * 100.0
                item = {"key": key, "value": normalized_score}
            else:
                item = {"key": key, "value": embedding.squeeze(0).cpu().numpy()}
            sv_result_list.append(item)
            if output_path is not None:
                embd_writer(key, embedding[0].cpu().numpy())
                if ref_embedding is not None:
                    if ref_embd_writer is None:
                        ref_embd_writer = WriteHelper(
                            "ark,scp:{}/ref_xvector.ark,{}/ref_xvector.scp".format(output_path, output_path)
                        )
                        score_writer = open(os.path.join(output_path, "score.txt"), "w")
                    ref_embd_writer(key, ref_embedding[0].cpu().numpy())
                    score_writer.write("{} {:.6f}\n".format(key, normalized_score))

        if output_path is not None:
            embd_writer.close()
            if ref_embd_writer is not None:
                ref_embd_writer.close()
                score_writer.close()

        return sv_result_list

    return _forward


def inference_launch(mode, **kwargs):
    if mode == "sv":
        return inference_sv(**kwargs)
    else:
        logging.info("Unknown decoding mode: {}".format(mode))
        return None


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Speaker Verification",
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
        required=False,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=True)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--vad_infer_config",
        type=str,
        help="VAD infer configuration",
    )
    group.add_argument(
        "--vad_model_file",
        type=str,
        help="VAD model parameter file",
    )
    group.add_argument(
        "--sv_train_config",
        type=str,
        help="ASR training configuration",
    )
    group.add_argument(
        "--sv_model_file",
        type=str,
        help="ASR model parameter file",
    )
    group.add_argument(
        "--cmvn_file",
        type=str,
        help="Global CMVN file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
             "*_file will be overwritten",
    )

    group = parser.add_argument_group("The inference configuration related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument(
        "--sv_threshold",
        type=float,
        default=0.9465,
        help="The threshold for verification"
    )
    parser.add_argument(
        "--embedding_node",
        type=str,
        default="resnet1_dense",
        help="The network node to extract embedding"
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    parser.add_argument(
        "--mode",
        type=str,
        default="sv",
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
