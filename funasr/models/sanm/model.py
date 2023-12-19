import logging

import torch

from funasr.models.transformer.model import Transformer
from funasr.utils.register import register_class, registry_tables

@register_class("model_classes", "SANM")
class SANM(Transformer):
	"""CTC-attention hybrid Encoder-Decoder model"""

	def __init__(
		self,
		*args,
		**kwargs,
	):

		super().__init__(*args, **kwargs)
