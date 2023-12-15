import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import tempfile
import codecs
import requests
import re
import copy
import torch
import torch.nn as nn
import random
import numpy as np
import time
# from funasr.layers.abs_normalize import AbsNormalize
from funasr.losses.label_smoothing_loss import (
	LabelSmoothingLoss,  # noqa: H301
)
# from funasr.models.ctc import CTC
# from funasr.models.decoder.abs_decoder import AbsDecoder
# from funasr.models.e2e_asr_common import ErrorCalculator
# from funasr.models.encoder.abs_encoder import AbsEncoder
# from funasr.models.frontend.abs_frontend import AbsFrontend
# from funasr.models.postencoder.abs_postencoder import AbsPostEncoder
from funasr.models.predictor.cif import mae_loss
# from funasr.models.preencoder.abs_preencoder import AbsPreEncoder
# from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.models.transformer.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.models.transformer.utils.nets_utils import th_accuracy
from funasr.train_utils.device_funcs import force_gatherable
# from funasr.models.base_model import FunASRModel
# from funasr.models.predictor.cif import CifPredictorV3
from funasr.models.paraformer.search import Hypothesis

from funasr.models.model_class_factory import *

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
	from torch.cuda.amp import autocast
else:
	# Nothing to do if torch<1.6.0
	@contextmanager
	def autocast(enabled=True):
		yield
from funasr.datasets.fun_datasets.load_audio_extract_fbank import load_audio, extract_fbank
from funasr.utils import postprocess_utils
from funasr.utils.datadir_writer import DatadirWriter

from funasr.models.transformer.model import Transformer

class Conformer(Transformer):
	"""CTC-attention hybrid Encoder-Decoder model"""

	
	def __init__(
		self,
		*args,
		**kwargs,
	):

		super().__init__(*args, **kwargs)
