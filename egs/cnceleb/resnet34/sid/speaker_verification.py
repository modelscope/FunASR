from __future__ import print_function
import numpy as np
import os
import kaldiio
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import math
from funasr.utils.types import str2triple_str
import logging
from typing import List, Union, Tuple, Sequence
from funasr.bin.sv_inference import inference_modelscope
import soundfile
import torch


class MultiProcessRunner:
    def __init__(self, fn):
        self.process = fn

    def run(self):
        parser = argparse.ArgumentParser("")
        # Task-independent options
        parser.add_argument("--njobs", type=int, default=16)
        parser.add_argument("--debug", action="store_true", default=False)
        parser.add_argument("--no_pbar", action="store_true", default=False)
        parser.add_argument("--verbose", action="store_true", default=False)
        parser.add_argument("--log_level", type=str, default="INFO")
        parser.add_argument("--sr", type=int, default=16000)

        task_list, shared_param, args = self.prepare(parser)
        chunk_size = int(math.ceil(float(len(task_list)) / args.njobs))
        if args.verbose:
            print("Split {} tasks into {} sub-tasks with chunk_size {}".format(len(task_list), args.njobs, chunk_size))
        subtask_list = [(i, task_list[i * chunk_size: (i + 1) * chunk_size], shared_param, args)
                        for i in range(args.njobs)]
        result_list = self.pool_run(subtask_list, args)
        self.post(result_list, args)

    def prepare(self, parser: argparse.ArgumentParser):
        raise NotImplementedError("Please implement the prepare function.")

    def post(self, results_list: list, args: argparse.Namespace):
        raise NotImplementedError("Please implement the post function.")

    def pool_run(self, tasks: list, args: argparse.Namespace):
        results = []
        if args.debug:
            one_result = self.process(tasks[0])
            results.append(one_result)
        else:
            pool = Pool(args.njobs)
            for one_result in tqdm(pool.imap(self.process, tasks), total=len(tasks), ascii=True, disable=args.no_pbar):
                results.append(one_result)
            pool.close()

        return results


class MyRunner(MultiProcessRunner):
    def prepare(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--gpu_inference",
            type=bool,
            default=False
        )
        parser.add_argument(
            "--data_path_and_name_and_type",
            type=str2triple_str,
            required=True,
            action="append"
        )
        parser.add_argument(
            "--gpu_devices",
            type=lambda devices: devices.split(","),
            default=None,
        )
        args = parser.parse_args()

        logging.basicConfig(
            level=args.log_level,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

        if args.gpu_inference and (args.gpu_devices is None or len(args.gpu_devices) == 0):
            logging.warning("gpu_inference is set to True, but gpu_devices is not given, use CPU instead.")
            args.gpu_inference = False

        if args.gpu_inference:
            args.njobs = args.njobs * len(args.gpu_devices)

        speech_dict = {}
        ref_speech_dict = {}
        for _path, _name, _type in args.data_path_and_name_and_type:
            if _name == "speech":
                speech_dict = self.read_data_path(_path)
            elif _name == "ref_speech":
                ref_speech_dict = self.read_data_path(_path)

        task_list, args.njobs = self.get_key_list(args.data_path_and_name_and_type, args.njobs)

        return task_list, [speech_dict, ref_speech_dict], args

    def read_data_path(self, file_path):
        results = {}
        for line in open(file_path, "r"):
            key, path = line.strip().split(" ", 1)
            results[key] = path

        return results

    def get_key_list(
            self,
            data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
            njobs: int
    ):
        first_data = data_path_and_name_and_type[0]
        content = open(first_data[0], "r").readlines()
        line_number = len(content)
        njobs = min(njobs, line_number)
        logging.warning("njobs is reduced to {}, since only {} lines exist in {}".format(
            njobs, line_number, first_data[0],
        ))
        key_list = [line.strip().split(" ", 1)[0] for line in content]
        return key_list, njobs

    def post(self, results_list: list, args: argparse.Namespace):
        for results in results_list:
            for key, value in results:
                logging.info("{} {}".format(key, value))


def process(task_args):
    task_id, key_list, [speech_dict, ref_speech_dict], args = task_args
    if args.gpu_inference:
        device = args.gpu_devices[task_id % len(args.gpu_devices)]
        torch.cuda.set_device("cuda:".format(device))
    inference_func = inference_modelscope(
        output_dir=None,
        batch_size=1,
        dtype="float32",
        ngpu=1 if args.gpu_inference else 0,
        seed=0,
        num_workers=0,
        log_level=logging.INFO,
        key_file=None,
        sv_train_config="sv.yaml",
        sv_model_file="sv.pb",
        model_tag=None,
        allow_variable_data_keys=True,
        streaming=False,
        embedding_node="resnet1_dense",
        sv_threshold=0.9465,
    )
    results = {}
    for key in key_list:
        speech = soundfile.read(speech_dict[key])[0]
        ref_speech = soundfile.read(ref_speech_dict[key])[0]
        ret = inference_func(None, (speech, ref_speech))
        results[key] = ret["value"]

    return results


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
