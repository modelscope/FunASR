import logging

from funasr.models.transformer.model import Transformer
from funasr.utils.register import register_class

@register_class("model_classes", "Branchformer")
class Branchformer(Transformer):
	"""CTC-attention hybrid Encoder-Decoder model"""

	def __init__(
		self,
		*args,
		**kwargs,
	):

		super().__init__(*args, **kwargs)
