from typing import Any
from typing import Dict
from typing import Union
from io import BytesIO

import logging
import torch
import torch.nn
import torch.optim


def filter_state_dict(
	dst_state: Dict[str, Union[float, torch.Tensor]],
	src_state: Dict[str, Union[float, torch.Tensor]],
):
	"""Filter name, size mismatch instances between dicts.

	Args:
		dst_state: reference state dict for filtering
		src_state: target state dict for filtering

	"""
	match_state = {}
	for key, value in src_state.items():
		if key in dst_state and (dst_state[key].size() == src_state[key].size()):
			match_state[key] = value
		else:
			if key not in dst_state:
				logging.warning(
					f"Filter out {key} from pretrained dict"
					+ " because of name not found in target dict"
				)
			else:
				logging.warning(
					f"Filter out {key} from pretrained dict"
					+ " because of size mismatch"
					+ f"({dst_state[key].size()}-{src_state[key].size()})"
				)
	return match_state

def assigment_scope_map(dst_state: dict, src_state: dict, scope_map: str=None):
	"""Compute the union of the current variables and checkpoint variables."""
	import collections
	import re

	# current model variables
	name_to_variable = collections.OrderedDict()
	for name, var in dst_state.items():
		name_to_variable[name] = var
	
	scope_map_num = 0
	if scope_map is not None:
		scope_map = scope_map.split(",")
		scope_map_num = len(scope_map) // 2
		for scope_map_idx in range(scope_map_num):
			scope_map_id = scope_map_idx * 2
			logging.info('assignment_map from scope {} to {}'.format(scope_map[scope_map_id], scope_map[scope_map_id+1]))
	
	assignment_map = {}
	for name, var in src_state.items():

		if scope_map:
			for scope_map_idx in range(scope_map_num):
				scope_map_id = scope_map_idx * 2
				try:
					idx = name.index(scope_map[scope_map_id])
					new_name = scope_map[scope_map_id+1] + name[idx + len(scope_map[scope_map_id]):]
					if new_name in name_to_variable:
						assignment_map[name] = var
				except:
					continue
		else:
			if name in name_to_variable:
				assignment_map[name] = var
	
	return assignment_map

def load_pretrained_model(
	path: str,
	model: torch.nn.Module,
	ignore_init_mismatch: bool,
	map_location: str = "cpu",
	oss_bucket=None,
	scope_map=None,
	excludes=None,
):
	"""Load a model state and set it to the model.

	Args:
		init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>

	Examples:

	"""
	
	obj = model
	
	if oss_bucket is None:
		src_state = torch.load(path, map_location=map_location)
	else:
		buffer = BytesIO(oss_bucket.get_object(path).read())
		src_state = torch.load(buffer, map_location=map_location)
	src_state = src_state["model"] if "model" in src_state else src_state
	
	if excludes is not None:
		for e in excludes.split(","):
			src_state = {k: v for k, v in src_state.items() if not k.startswith(e)}
	
	dst_state = obj.state_dict()
	src_state = assigment_scope_map(dst_state, src_state, scope_map)
	
	if ignore_init_mismatch:
		src_state = filter_state_dict(dst_state, src_state)
	
	logging.debug("Loaded src_state keys: {}".format(src_state.keys()))
	logging.debug("Loaded dst_state keys: {}".format(dst_state.keys()))
	dst_state.update(src_state)
	obj.load_state_dict(dst_state, strict=True)
