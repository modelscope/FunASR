import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types
from funasr.models.transformer.utils.nets_utils import make_pad_mask


def sense_voice_encode(
	self,
	x: Tensor,
	ilens: torch.Tensor = None,
	**kwargs,
):
	
	x = F.gelu(self.encoder.conv1(input))
	x = F.gelu(self.encoder.conv2(x))
	x = x.permute(0, 2, 1)
	
	n_frames = x.size(1)
	max_pos = self.encoder.positional_embedding.size(0)
	max_pos = n_frames if n_frames > max_pos else max_pos
	x = (x[:, :max_pos, :] + self.encoder.positional_embeddingx[:, :max_pos, :]).to(x.dtype)
	
	
	if ilens is not None:
		if self.downsample_rate == 4:
			olens = (
				1
				+ (
					ilens
					- self.encoder.conv1.kernel_size[0]
					+ 2 * self.encoder.conv1.padding[0]
				)
				// self.encoder.conv1.stride[0]
			)
		else:
			olens = ilens
		olens = (
			1
			+ (
				olens
				- self.encoder.conv2.kernel_size[0]
				+ 2 * self.encoder.conv2.padding[0]
			)
			// self.encoder.conv2.stride[0]
		)
		olens = torch.clamp(olens, max=max_pos)
	else:
		olens = None
	
	if self.use_padmask and olens is not None:
		padding_mask = (~make_pad_mask(olens)[:, None, :]).to(x.device)
	else:
		padding_mask = None
	
	x = self.dropout(x)
	
	for layer, block in enumerate(self.encoder.blocks):
		x = block(x, mask=padding_mask, is_pad_mask=True)
		

	x = self.encoder.ln_post(x)
	
	if ilens is None:
		return x
	else:
		return x, olens
