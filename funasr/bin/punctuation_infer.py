#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
import os
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Dict
from typing import Any
from typing import List

import numpy as np
import torch
from torch.nn.parallel import data_parallel
from typeguard import check_argument_types

from funasr.datasets.preprocessor import CommonPreprocessor
from funasr.utils.cli_utils import get_commandline_args
from funasr.fileio.datadir_writer import DatadirWriter
from funasr.tasks.punctuation import PunctuationTask
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.forward_adaptor import ForwardAdaptor
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import config_argparse
from funasr.utils.types import float_or_none
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none
from funasr.punctuation.text_preprocessor import split_words, split_to_mini_sentence


def inference(
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    output_dir: str,
    log_level: Union[int, str],
    train_config: Optional[str],
    model_file: Optional[str],
    key_file: Optional[str] = None,
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
    raw_inputs: Union[List[Any], bytes, str] = None,
    
    **kwargs,
):
    inference_pipeline = inference_modelscope(
        output_dir=output_dir,
        raw_inputs=raw_inputs,
        batch_size=batch_size,
        dtype=dtype,
        ngpu=ngpu,
        seed=seed,
        num_workers=num_workers,
        log_level=log_level,
        key_file=key_file,
        train_config=train_config,
        model_file=model_file,
        **kwargs,
    )
    return inference_pipeline(data_path_and_name_and_type, raw_inputs)


def inference_modelscope(
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

    # 2. Build Model
    model, train_args = PunctuationTask.build_model_from_file(
        train_config, model_file, device)
    # Wrape model to make model.nll() data-parallel
    wrapped_model = ForwardAdaptor(model, "inference")
    wrapped_model.to(dtype=getattr(torch, dtype)).to(device=device).eval()
    logging.info(f"Model:\n{model}")
    punc_list = train_args.punc_list
    period = 0
    for i in range(len(punc_list)):
        if punc_list[i] == ",":
            punc_list[i] = "，"
        elif punc_list[i] == "?":
            punc_list[i] = "？"
        elif punc_list[i] == "。":
            period = i

    preprocessor = CommonPreprocessor(
        train=False,
        token_type="word",
        token_list=train_args.token_list,
        bpemodel=train_args.bpemodel,
        text_cleaner=train_args.cleaner,
        g2p_type=train_args.g2p,
        text_name="text",
        non_linguistic_symbols=train_args.non_linguistic_symbols,
    )

    print("start decoding!!!")

    def _forward(
        data_path_and_name_and_type,
        raw_inputs: Union[List[Any], bytes, str] = None,
        output_dir_v2: Optional[str] = None,
    ):
        results = []
        split_size = 20

        if raw_inputs != None:
            line = raw_inputs.strip()
            key = "demo"
            if line=="":
                item = {'key': key, 'value': ""}
                results.append(item)
                return results
            cache_sent = []
            words = split_words(line)
            new_mini_sentence = ""
            new_mini_sentence_punc = ""
            cache_pop_trigger_limit = 200
            mini_sentences = split_to_mini_sentence(words, split_size)
            for mini_sentence_i in range(len(mini_sentences)):
                mini_sentence = mini_sentences[mini_sentence_i]
                mini_sentence = cache_sent + mini_sentence
                data = {"text": " ".join(mini_sentence)}
                batch = preprocessor(data=data, uid="12938712838719")
                batch["text_lengths"] = torch.from_numpy(
                    np.array([len(batch["text"])], dtype='int32'))
                batch["text"] = torch.from_numpy(batch["text"])
                # Extend one dimension to fake a batch dim.
                batch["text"] = torch.unsqueeze(batch["text"], 0)
                batch = to_device(batch, device)
                y, _ = wrapped_model(**batch)
                _, indices = y.view(-1, y.shape[-1]).topk(1, dim=1)
                punctuations = indices
                if indices.size()[0] != 1:
                    punctuations = torch.squeeze(indices)
                assert punctuations.size()[0] == len(mini_sentence)
    
                # Search for the last Period/QuestionMark as cache 
                if mini_sentence_i < len(mini_sentences)-1:
                    sentenceEnd = -1
                    last_comma_index = -1
                    for i in range(len(punctuations)-2,1,-1):
                        if punc_list[punctuations[i]] == "。" or punc_list[punctuations[i]] == "？":
                            sentenceEnd = i
                            break
                        if last_comma_index < 0 and punc_list[punctuations[i]] == "，":
                            last_comma_index = i
                    if sentenceEnd < 0 and len(mini_sentence) > cache_pop_trigger_limit and last_comma_index >= 0:
                        # The sentence it too long, cut off at a comma.
                        sentenceEnd = last_comma_index
                        punctuations[sentenceEnd] = period
                    cache_sent = mini_sentence[sentenceEnd+1:]
                    mini_sentence = mini_sentence[0:sentenceEnd+1]
                    punctuations = punctuations[0:sentenceEnd+1]
    
                punctuations_np = punctuations.cpu().numpy()
                new_mini_sentence_punc += "".join([str(x) for x in punctuations_np])
                words_with_punc = []
                for i in range(len(mini_sentence)):
                    if i>0:
                        if len(mini_sentence[i][0].encode()) == 1 and len(mini_sentence[i-1][0].encode()) == 1:
                            mini_sentence[i] = " "+ mini_sentence[i]
                    words_with_punc.append(mini_sentence[i])
                    if punc_list[punctuations[i]] != "_":
                        words_with_punc.append(punc_list[punctuations[i]])
                new_mini_sentence += "".join(words_with_punc)
     
                # Add Period for the end of the sentence
                new_mini_sentence_out = new_mini_sentence
                new_mini_sentence_punc_out = new_mini_sentence_punc
                if mini_sentence_i == len(mini_sentences)-1:
                    if new_mini_sentence[-1]=="，" or new_mini_sentence[-1]=="、":
                        new_mini_sentence_out = new_mini_sentence[:-1] + "。"
                        new_mini_sentence_punc_out  = new_mini_sentence_punc[:-1] + str(period)
                    elif new_mini_sentence[-1]!="。" and new_mini_sentence[-1]!="？":
                        new_mini_sentence_out=new_mini_sentence+"。"
                        new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + str(period)
                    item = {'key': key, 'value': new_mini_sentence_out}
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
                    cache_sent = []
                    words = split_words(segs[1])
                    new_mini_sentence = ""
                    new_mini_sentence_punc = ""
                    cache_pop_trigger_limit = 200
                    mini_sentences = split_to_mini_sentence(words, split_size)
                    for mini_sentence_i in range(len(mini_sentences)):
                        mini_sentence = mini_sentences[mini_sentence_i]
                        mini_sentence = cache_sent + mini_sentence
                        data = {"text": " ".join(mini_sentence)}
                        batch = preprocessor(data=data, uid="12938712838719")
                        batch["text_lengths"] = torch.from_numpy(
                            np.array([len(batch["text"])], dtype='int32'))
                        batch["text"] = torch.from_numpy(batch["text"])
                        # Extend one dimension to fake a batch dim.
                        batch["text"] = torch.unsqueeze(batch["text"], 0)
                        batch = to_device(batch, device)
                        y, _ = wrapped_model(**batch)
                        _, indices = y.view(-1, y.shape[-1]).topk(1, dim=1)
                        punctuations = indices
                        if indices.size()[0] != 1:
                            punctuations = torch.squeeze(indices)
                        assert punctuations.size()[0] == len(mini_sentence)
    
                        # Search for the last Period/QuestionMark as cache 
                        if mini_sentence_i < len(mini_sentences)-1:
                            sentenceEnd = -1
                            last_comma_index = -1
                            for i in range(len(punctuations)-2,1,-1):
                                if punc_list[punctuations[i]] == "。" or punc_list[punctuations[i]] == "？":
                                    sentenceEnd = i
                                    break
                                if last_comma_index < 0 and punc_list[punctuations[i]] == "，":
                                    last_comma_index = i
                            if sentenceEnd < 0 and len(mini_sentence) > cache_pop_trigger_limit and last_comma_index >= 0:
                                # The sentence it too long, cut off at a comma.
                                sentenceEnd = last_comma_index
                                punctuations[sentenceEnd] = period
                            cache_sent = mini_sentence[sentenceEnd+1:]
                            mini_sentence = mini_sentence[0:sentenceEnd+1]
                            punctuations = punctuations[0:sentenceEnd+1]
    
                        punctuations_np = punctuations.cpu().numpy()
                        new_mini_sentence_punc += "".join([str(x) for x in punctuations_np])
                        words_with_punc = []
                        for i in range(len(mini_sentence)):
                            if i>0:
                                if len(mini_sentence[i][0].encode()) == 1 and len(mini_sentence[i-1][0].encode()) == 1:
                                    mini_sentence[i] = " "+ mini_sentence[i]
                            words_with_punc.append(mini_sentence[i])
                            if punc_list[punctuations[i]] != "_":
                                words_with_punc.append(punc_list[punctuations[i]])
                        new_mini_sentence += "".join(words_with_punc)
     
                        # Add Period for the end of the sentence
                        new_mini_sentence_out = new_mini_sentence
                        new_mini_sentence_punc_out = new_mini_sentence_punc
                        if mini_sentence_i == len(mini_sentences)-1:
                            if new_mini_sentence[-1]=="，" or new_mini_sentence[-1]=="、":
                                new_mini_sentence_out = new_mini_sentence[:-1] + "。"
                                new_mini_sentence_punc_out  = new_mini_sentence_punc[:-1] + str(period)
                            elif new_mini_sentence[-1]!="。" and new_mini_sentence[-1]!="？":
                                new_mini_sentence_out=new_mini_sentence+"。"
                                new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + str(period)
                            item = {'key': key, 'value': new_mini_sentence_out}
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

    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
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


    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--train_config", type=str)
    group.add_argument("--model_file", type=str)

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
   # kwargs.pop("config", None)
    inference(**kwargs)

if __name__ == "__main__":
    main()



