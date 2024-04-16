import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types
from funasr.models.transformer.utils.nets_utils import make_pad_mask

def sense_voice_decode(
	self,
	x: Tensor,
	xa: Tensor,
	kv_cache: Optional[dict] = None,
	**kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Forward decoder.

	Args:
		hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
		hlens: (batch)
		ys_in_pad:
			input token ids, int64 (batch, maxlen_out)
			if input_layer == "embed"
			input tensor (batch, maxlen_out, #mels) in the other cases
		ys_in_lens: (batch)
	Returns:
		(tuple): tuple containing:

		x: decoded token score before softmax (batch, maxlen_out, token)
			if use_output_layer is True,
		olens: (batch, )
	"""
	
	hlens = kwargs.get("hlens", None)
	
	ys_in_lens = kwargs.get("ys_in_lens", None)
	
	offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
	tgt, memory = x, xa
	tgt = (
		self.decoder.token_embedding(tgt)
		+ self.decoder.positional_embedding[offset : offset + tgt.size(1)]
	)
	# tgt = self.dropout(tgt)
	
	x = tgt.to(memory.dtype)
	
	if self.use_padmask:
		memory_mask = (~make_pad_mask(hlens)[:, None, :]).to(memory.device)
	else:
		memory_mask = None
	
	for layer, block in enumerate(self.decoder.blocks):
		x = block(x, memory, mask=self.decoder.mask, memory_mask=memory_mask, is_pad_mask=False, is_pad_memory_mask=True)


	x = self.decoder.ln(x)
	x = (
		x @ torch.transpose(self.decoder.token_embedding.weight.to(x.dtype), 0, 1)
	).float()
	
	
	return x
	