#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import argparse
import logging
import os
import sys
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch.nn.parallel import data_parallel

from funasr.build_utils.build_model_from_file import build_model_from_file
from funasr.build_utils.build_streaming_iterator import build_streaming_iterator
from funasr.datasets.preprocessor import LMPreprocessor
from funasr.fileio.datadir_writer import DatadirWriter
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.forward_adaptor import ForwardAdaptor
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.types import float_or_none
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none


def inference_lm(
        batch_size: int,
        dtype: str,
        ngpu: int,
        seed: int,
        num_workers: int,
        log_level: Union[int, str],
        key_file: Optional[str],
        train_config: Optional[str],
        model_file: Optional[str],
        log_base: Optional[float] = 10,
        allow_variable_data_keys: bool = False,
        split_with_space: Optional[bool] = False,
        seg_dict_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        param_dict: dict = None,
        **kwargs,
):
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build Model
    model, train_args = build_model_from_file(
        train_config, model_file, None, device, "lm")
    wrapped_model = ForwardAdaptor(model, "nll")
    wrapped_model.to(dtype=getattr(torch, dtype)).to(device=device).eval()
    logging.info(f"Model:\n{model}")

    preprocessor = LMPreprocessor(
        train=False,
        token_type=train_args.token_type,
        token_list=train_args.token_list,
        bpemodel=train_args.bpemodel,
        text_cleaner=train_args.cleaner,
        g2p_type=train_args.g2p,
        text_name="text",
        non_linguistic_symbols=train_args.non_linguistic_symbols,
        split_with_space=split_with_space,
        seg_dict_file=seg_dict_file
    )

    def _forward(
            data_path_and_name_and_type,
            raw_inputs: Union[List[Any], bytes, str] = None,
            output_dir_v2: Optional[str] = None,
            param_dict: dict = None,
    ):
        results = []
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            writer = DatadirWriter(output_path)
        else:
            writer = None

        if raw_inputs != None:
            line = raw_inputs.strip()
            key = "lm demo"
            if line == "":
                item = {'key': key, 'value': ""}
                results.append(item)
                return results
            batch = {}
            batch['text'] = line
            if preprocessor != None:
                batch = preprocessor(key, batch)

            #  Force data-precision
            for name in batch:
                value = batch[name]
                if not isinstance(value, np.ndarray):
                    raise RuntimeError(
                        f"All values must be converted to np.ndarray object "
                        f'by preprocessing, but "{name}" is still {type(value)}.'
                    )
                # Cast to desired type
                if value.dtype.kind == "f":
                    value = value.astype("float32")
                elif value.dtype.kind == "i":
                    value = value.astype("long")
                else:
                    raise NotImplementedError(f"Not supported dtype: {value.dtype}")
                batch[name] = value

            batch["text_lengths"] = torch.from_numpy(
                np.array([len(batch["text"])], dtype='int32'))
            batch["text"] = np.expand_dims(batch["text"], axis=0)

            with torch.no_grad():
                batch = to_device(batch, device)
                if ngpu <= 1:
                    nll, lengths = wrapped_model(**batch)
                else:
                    nll, lengths = data_parallel(
                        wrapped_model, (), range(ngpu), module_kwargs=batch
                    )
                ## compute ppl
                ppl_out_batch = ""
                ids2tokens = preprocessor.token_id_converter.ids2tokens
                for sent_ids, sent_nll in zip(batch['text'], nll):
                    pre_word = "<s>"
                    cur_word = None
                    sent_lst = ids2tokens(sent_ids) + ['</s>']
                    ppl_out = " ".join(sent_lst) + "\n"
                    for word, word_nll in zip(sent_lst, sent_nll):
                        cur_word = word
                        word_nll = -word_nll.cpu()
                        if log_base is None:
                            word_prob = np.exp(word_nll)
                        else:
                            word_prob = log_base ** (word_nll / np.log(log_base))
                        ppl_out += '    p( {cur} | {pre} ) = {prob} [ {word_nll} ]\n'.format(
                            cur=cur_word,
                            pre=pre_word,
                            prob=round(word_prob.item(), 8),
                            word_nll=round(word_nll.item(), 8)
                        )
                        pre_word = cur_word

                    sent_nll_mean = sent_nll.mean().cpu().numpy()
                    sent_nll_sum = sent_nll.sum().cpu().numpy()
                    if log_base is None:
                        sent_ppl = np.exp(sent_nll_mean)
                    else:
                        sent_ppl = log_base ** (sent_nll_mean / np.log(log_base))
                    ppl_out += 'logprob= {sent_nll} ppl= {sent_ppl}\n\n'.format(
                        sent_nll=round(-sent_nll_sum.item(), 4),
                        sent_ppl=round(sent_ppl.item(), 4)
                    )
                    ppl_out_batch += ppl_out
                    item = {'key': key, 'value': ppl_out}
                    if writer is not None:
                        writer["ppl"][key + ":\n"] = ppl_out
                    results.append(item)

            return results

        # 3. Build data-iterator
        loader = build_streaming_iterator(
            task_name="lm",
            preprocess_args=train_args,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
        )

        # 4. Start for-loop
        total_nll = 0.0
        total_ntokens = 0
        ppl_out_all = ""
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            ppl_out_batch = ""
            with torch.no_grad():
                batch = to_device(batch, device)
                if ngpu <= 1:
                    # NOTE(kamo): data_parallel also should work with ngpu=1,
                    # but for debuggability it's better to keep this block.
                    nll, lengths = wrapped_model(**batch)
                else:
                    nll, lengths = data_parallel(
                        wrapped_model, (), range(ngpu), module_kwargs=batch
                    )
                ## print ppl
                ids2tokens = preprocessor.token_id_converter.ids2tokens
                for key, sent_ids, sent_nll in zip(keys, batch['text'], nll):
                    pre_word = "<s>"
                    cur_word = None
                    sent_lst = ids2tokens(sent_ids) + ['</s>']
                    ppl_out = " ".join(sent_lst) + "\n"
                    for word, word_nll in zip(sent_lst, sent_nll):
                        cur_word = word
                        word_nll = -word_nll.cpu()
                        if log_base is None:
                            word_prob = np.exp(word_nll)
                        else:
                            word_prob = log_base ** (word_nll / np.log(log_base))
                        ppl_out += '    p( {cur} | {pre} ) = {prob} [ {word_nll} ]\n'.format(
                            cur=cur_word,
                            pre=pre_word,
                            prob=round(word_prob.item(), 8),
                            word_nll=round(word_nll.item(), 8)
                        )
                        pre_word = cur_word

                    sent_nll_mean = sent_nll.mean().cpu().numpy()
                    sent_nll_sum = sent_nll.sum().cpu().numpy()
                    if log_base is None:
                        sent_ppl = np.exp(sent_nll_mean)
                    else:
                        sent_ppl = log_base ** (sent_nll_mean / np.log(log_base))
                    ppl_out += 'logprob= {sent_nll} ppl= {sent_ppl}\n\n'.format(
                        sent_nll=round(-sent_nll_sum.item(), 4),
                        sent_ppl=round(sent_ppl.item(), 4)
                    )
                    ppl_out_batch += ppl_out
                    utt2nll = round(-sent_nll_sum.item(), 5)
                    item = {'key': key, 'value': ppl_out}
                    if writer is not None:
                        writer["ppl"][key + ":\n"] = ppl_out
                        writer["utt2nll"][key] = str(utt2nll)
                    results.append(item)

            ppl_out_all += ppl_out_batch

            assert _bs == len(nll) == len(lengths), (_bs, len(nll), len(lengths))
            # nll: (B, L) -> (B,)
            nll = nll.detach().cpu().numpy().sum(1)
            # lengths: (B,)
            lengths = lengths.detach().cpu().numpy()
            total_nll += nll.sum()
            total_ntokens += lengths.sum()

        if log_base is None:
            ppl = np.exp(total_nll / total_ntokens)
        else:
            ppl = log_base ** (total_nll / total_ntokens / np.log(log_base))

        avg_ppl = 'logprob= {total_nll} ppl= {total_ppl}\n'.format(
            total_nll=round(-total_nll.item(), 4),
            total_ppl=round(ppl.item(), 4)
        )
        item = {'key': 'AVG PPL', 'value': avg_ppl}
        ppl_out_all += avg_ppl
        if writer is not None:
            writer["ppl"]["AVG PPL : "] = avg_ppl
        results.append(item)

        return results

    return _forward


def inference_launch(mode, **kwargs):
    if mode == "transformer":
        return inference_lm(**kwargs)
    else:
        logging.info("Unknown decoding mode: {}".format(mode))
        return None


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
