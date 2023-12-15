import os
import torch
from torch.nn import functional as F
import yaml
import numpy as np

def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
	if maxlen is None:
		maxlen = lengths.max()
	row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
	matrix = torch.unsqueeze(lengths, dim=-1)
	mask = row_vector < matrix
	mask = mask.detach()

	return mask.type(dtype).to(device) if device is not None else mask.type(dtype)

def apply_cmvn(inputs, mvn):
	device = inputs.device
	dtype = inputs.dtype
	frame, dim = inputs.shape
	meams = np.tile(mvn[0:1, :dim], (frame, 1))
	vars = np.tile(mvn[1:2, :dim], (frame, 1))
	inputs -= torch.from_numpy(meams).type(dtype).to(device)
	inputs *= torch.from_numpy(vars).type(dtype).to(device)

	return inputs.type(torch.float32)




def drop_and_add(inputs: torch.Tensor,
                 outputs: torch.Tensor,
                 training: bool,
                 dropout_rate: float = 0.1,
                 stoch_layer_coeff: float = 1.0):



	outputs = F.dropout(outputs, p=dropout_rate, training=training, inplace=True)
	outputs *= stoch_layer_coeff

	input_dim = inputs.size(-1)
	output_dim = outputs.size(-1)

	if input_dim == output_dim:
		outputs += inputs
	return outputs


def proc_tf_vocab(vocab_path):
	with open(vocab_path, encoding="utf-8") as f:
		token_list = [line.rstrip() for line in f]
		if '<unk>' not in token_list:
			token_list.append('<unk>')
	return token_list


def gen_config_for_tfmodel(config_path, vocab_path, output_dir):
	token_list = proc_tf_vocab(vocab_path)
	with open(config_path, encoding="utf-8") as f:
		config = yaml.safe_load(f)
	
	config['token_list'] = token_list
	
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
		yaml_no_alias_safe_dump(config, f, indent=4, sort_keys=False)


class NoAliasSafeDumper(yaml.SafeDumper):
	# Disable anchor/alias in yaml because looks ugly
	def ignore_aliases(self, data):
		return True


def yaml_no_alias_safe_dump(data, stream=None, **kwargs):
	"""Safe-dump in yaml with no anchor/alias"""
	return yaml.dump(
		data, stream, allow_unicode=True, Dumper=NoAliasSafeDumper, **kwargs
	)


if __name__ == '__main__':
	import sys
	
	config_path = sys.argv[1]
	vocab_path = sys.argv[2]
	output_dir = sys.argv[3]
	gen_config_for_tfmodel(config_path, vocab_path, output_dir)