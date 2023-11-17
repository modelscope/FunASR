#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os

import logging
import torch
import numpy as np
from funasr.utils.download_and_prepare_model import prepare_model

from funasr.utils.types import str2bool

def infer(task_name: str = "asr",
          model: str = None,
          # mode: str = None,
          vad_model: str = None,
          disable_vad: bool = False,
          punc_model: str = None,
          disable_punc: bool = False,
          model_hub: str = "ms",
          cache_dir: str = None,
          **kwargs,
          ):

	# set logging messages
	logging.basicConfig(
		level=logging.ERROR,
	)

	model, vad_model, punc_model, kwargs = prepare_model(model, vad_model, punc_model, model_hub, cache_dir, **kwargs)
	if task_name == "asr":
		from funasr.bin.asr_inference_launch import inference_launch
		
		inference_pipeline = inference_launch(**kwargs)
	elif task_name == "":
		pipeline = 1
	elif task_name == "":
		pipeline = 2
	elif task_name == "":
		pipeline = 2
	
	def _infer_fn(input, **kwargs):
		data_type = kwargs.get('data_type', 'sound')
		data_path_and_name_and_type = [input, 'speech', data_type]
		raw_inputs = None
		if isinstance(input, torch.Tensor):
			input = input.numpy()
		if isinstance(input, np.ndarray):
			data_path_and_name_and_type = None
			raw_inputs = input
		
		return inference_pipeline(data_path_and_name_and_type, raw_inputs=raw_inputs, **kwargs)
	
	return _infer_fn


def main(cmd=None):
	# print(get_commandline_args(), file=sys.stderr)
	from funasr.bin.argument import get_parser
	
	parser = get_parser()
	parser.add_argument('input', help='input file to transcribe')
	parser.add_argument(
	    "--task_name",
	    type=str,
	    default="asr",
	    help="The decoding mode",
	)
	parser.add_argument(
		"-m",
	    "--model",
	    type=str,
	    default="paraformer-zh",
	    help="The asr mode name",
	)
	parser.add_argument(
		"-v",
	    "--vad_model",
	    type=str,
	    default="fsmn-vad",
	    help="vad model name",
	)
	parser.add_argument(
		"-dv",
	    "--disable_vad",
	    type=str2bool,
	    default=False,
	    help="",
	)
	parser.add_argument(
		"-p",
	    "--punc_model",
	    type=str,
	    default="ct-punc",
	    help="",
	)
	parser.add_argument(
		"-dp",
	    "--disable_punc",
	    type=str2bool,
	    default=False,
	    help="",
	)
	parser.add_argument(
	    "--batch_size_token",
	    type=int,
	    default=5000,
	    help="",
	)
	parser.add_argument(
	    "--batch_size_token_threshold_s",
	    type=int,
	    default=35,
	    help="",
	)
	parser.add_argument(
	    "--max_single_segment_time",
	    type=int,
	    default=5000,
	    help="",
	)
	args = parser.parse_args(cmd)
	kwargs = vars(args)
	
	# set logging messages
	logging.basicConfig(
		level=logging.ERROR,
	)
	logging.info("Decoding args: {}".format(kwargs))
	
	# kwargs["ncpu"] = 2 #os.cpu_count()
	kwargs.pop("data_path_and_name_and_type")
	print("args: {}".format(kwargs))
	p = infer(**kwargs)
	
	res = p(**kwargs)
	print(res)
