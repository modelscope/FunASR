import torch
from torch.nn import functional as F

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

