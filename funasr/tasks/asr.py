import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr.datasets.collate_fn import CommonCollateFn
from funasr.datasets.preprocessor import CommonPreprocessor
from funasr.models.ctc import CTC
from funasr.models.decoder.abs_decoder import AbsDecoder
from funasr.models.decoder.rnn_decoder import RNNDecoder
from funasr.models.decoder.transformer_decoder import (
	DynamicConvolution2DTransformerDecoder,  # noqa: H301
)
from funasr.models.decoder.transformer_decoder import DynamicConvolutionTransformerDecoder
from funasr.models.decoder.transformer_decoder import (
	LightweightConvolution2DTransformerDecoder,  # noqa: H301
)
from funasr.models.decoder.transformer_decoder import (
	LightweightConvolutionTransformerDecoder,  # noqa: H301
)
from funasr.models.decoder.transformer_decoder import TransformerDecoder
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.encoder.conformer_encoder import ConformerEncoder
from funasr.models.encoder.rnn_encoder import RNNEncoder
from funasr.models.encoder.transformer_encoder import TransformerEncoder
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.frontend.default import DefaultFrontend
from funasr.models.frontend.fused import FusedFrontends
from funasr.models.frontend.s3prl import S3prlFrontend
from funasr.models.frontend.windowing import SlidingWindow
from funasr.models.postencoder.abs_postencoder import AbsPostEncoder
from funasr.models.postencoder.hugging_face_transformers_postencoder import (
	HuggingFaceTransformersPostEncoder,  # noqa: H301
)
from funasr.models.preencoder.abs_preencoder import AbsPreEncoder
from funasr.models.preencoder.linear import LinearProjection
from funasr.models.preencoder.sinc import LightweightSincConvs
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.models.specaug.specaug import SpecAug
from funasr.layers.abs_normalize import AbsNormalize
from funasr.layers.global_mvn import GlobalMVN
from funasr.layers.utterance_mvn import UtteranceMVN
from funasr.tasks.abs_task import AbsTask
from funasr.text.phoneme_tokenizer import g2p_choices
from funasr.torch_utils.initialize import initialize
from funasr.train.abs_espnet_model import AbsESPnetModel
from funasr.train.class_choices import ClassChoices
from funasr.train.trainer import Trainer
from funasr.utils.get_default_kwargs import get_default_kwargs
from funasr.utils.nested_dict_action import NestedDictAction
from funasr.utils.types import float_or_none
from funasr.utils.types import int_or_none
from funasr.utils.types import str2bool
from funasr.utils.types import str_or_none

from funasr.models.specaug.specaug import SpecAugLFR
from funasr.models.predictor.cif import CifPredictor, CifPredictorV2
from funasr.modules.subsampling import Conv1dSubsampling
from funasr.models.e2e_asr import ESPnetASRModel
from funasr.models.e2e_uni_asr import UniASR
from funasr.models.encoder.sanm_encoder import SANMEncoder, SANMEncoderChunkOpt
from funasr.models.decoder.sanm_decoder import ParaformerSANMDecoder, FsmnDecoderSCAMAOpt
from funasr.models.e2e_asr_paraformer import Paraformer, ParaformerBert
from funasr.models.decoder.transformer_decoder import ParaformerDecoderSAN

frontend_choices = ClassChoices(
	name="frontend",
	classes=dict(
		default=DefaultFrontend,
		sliding_window=SlidingWindow,
		s3prl=S3prlFrontend,
		fused=FusedFrontends,
	),
	type_check=AbsFrontend,
	default="default",
)
specaug_choices = ClassChoices(
	name="specaug",
	classes=dict(
		specaug=SpecAug,
		specaug_lfr=SpecAugLFR,
	),
	type_check=AbsSpecAug,
	default=None,
	optional=True,
)
normalize_choices = ClassChoices(
	"normalize",
	classes=dict(
		global_mvn=GlobalMVN,
		utterance_mvn=UtteranceMVN,
	),
	type_check=AbsNormalize,
	default=None,
	optional=True,
)
model_choices = ClassChoices(
	"model",
	classes=dict(
		asr=ESPnetASRModel,
		uniasr=UniASR,
		paraformer=Paraformer,
		paraformer_bert=ParaformerBert,
	),
	type_check=AbsESPnetModel,
	default="asr",
)
preencoder_choices = ClassChoices(
	name="preencoder",
	classes=dict(
		sinc=LightweightSincConvs,
		linear=LinearProjection,
	),
	type_check=AbsPreEncoder,
	default=None,
	optional=True,
)
encoder_choices = ClassChoices(
	"encoder",
	classes=dict(
		conformer=ConformerEncoder,
		transformer=TransformerEncoder,
		rnn=RNNEncoder,
		sanm=SANMEncoder,
		sanm_chunk_opt=SANMEncoderChunkOpt,
	),
	type_check=AbsEncoder,
	default="rnn",
)
encoder_choices2 = ClassChoices(
	"encoder2",
	classes=dict(
		conformer=ConformerEncoder,
		transformer=TransformerEncoder,
		rnn=RNNEncoder,
		sanm=SANMEncoder,
		sanm_chunk_opt=SANMEncoderChunkOpt,
	),
	type_check=AbsEncoder,
	default="rnn",
)
postencoder_choices = ClassChoices(
	name="postencoder",
	classes=dict(
		hugging_face_transformers=HuggingFaceTransformersPostEncoder,
	),
	type_check=AbsPostEncoder,
	default=None,
	optional=True,
)
decoder_choices = ClassChoices(
	"decoder",
	classes=dict(
		transformer=TransformerDecoder,
		lightweight_conv=LightweightConvolutionTransformerDecoder,
		lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
		dynamic_conv=DynamicConvolutionTransformerDecoder,
		dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
		rnn=RNNDecoder,
		fsmn_scama_opt=FsmnDecoderSCAMAOpt,
		paraformer_decoder_sanm=ParaformerSANMDecoder,
		paraformer_decoder_san=ParaformerDecoderSAN,
	),
	type_check=AbsDecoder,
	default="rnn",
)
decoder_choices2 = ClassChoices(
	"decoder2",
	classes=dict(
		transformer=TransformerDecoder,
		lightweight_conv=LightweightConvolutionTransformerDecoder,
		lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
		dynamic_conv=DynamicConvolutionTransformerDecoder,
		dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
		rnn=RNNDecoder,
		fsmn_scama_opt=FsmnDecoderSCAMAOpt,
		paraformer_decoder_sanm=ParaformerSANMDecoder,
	),
	type_check=AbsDecoder,
	default="rnn",
)
predictor_choices = ClassChoices(
	name="predictor",
	classes=dict(
		cif_predictor=CifPredictor,
		ctc_predictor=None,
		cif_predictor_v2=CifPredictorV2,
	),
	type_check=None,
	default="cif_predictor",
	optional=True,
)
predictor_choices2 = ClassChoices(
	name="predictor2",
	classes=dict(
		cif_predictor=CifPredictor,
		ctc_predictor=None,
		cif_predictor_v2=CifPredictorV2,
	),
	type_check=None,
	default="cif_predictor",
	optional=True,
)
stride_conv_choices = ClassChoices(
	name="stride_conv",
	classes=dict(
		stride_conv1d=Conv1dSubsampling
	),
	type_check=None,
	default="stride_conv1d",
	optional=True,
)


class ASRTask(AbsTask):
	# If you need more than one optimizers, change this value
	num_optimizers: int = 1

	# Add variable objects configurations
	class_choices_list = [
		# --frontend and --frontend_conf
		frontend_choices,
		# --specaug and --specaug_conf
		specaug_choices,
		# --normalize and --normalize_conf
		normalize_choices,
		# --model and --model_conf
		model_choices,
		# --preencoder and --preencoder_conf
		preencoder_choices,
		# --encoder and --encoder_conf
		encoder_choices,
		# --postencoder and --postencoder_conf
		postencoder_choices,
		# --decoder and --decoder_conf
		decoder_choices,
	]

	# If you need to modify train() or eval() procedures, change Trainer class here
	trainer = Trainer

	@classmethod
	def add_task_arguments(cls, parser: argparse.ArgumentParser):
		group = parser.add_argument_group(description="Task related")

		# NOTE(kamo): add_arguments(..., required=True) can't be used
		# to provide --print_config mode. Instead of it, do as
		required = parser.get_default("required")
		required += ["token_list"]

		group.add_argument(
			"--token_list",
			type=str_or_none,
			default=None,
			help="A text mapping int-id to token",
		)
		group.add_argument(
			"--split_with_space",
			type=str2bool,
			default=True,
			help="whether to split text using <space>",
		)
		group.add_argument(
			"--init",
			type=lambda x: str_or_none(x.lower()),
			default=None,
			help="The initialization method",
			choices=[
				"chainer",
				"xavier_uniform",
				"xavier_normal",
				"kaiming_uniform",
				"kaiming_normal",
				None,
			],
		)

		group.add_argument(
			"--input_size",
			type=int_or_none,
			default=None,
			help="The number of input dimension of the feature",
		)

		group.add_argument(
			"--ctc_conf",
			action=NestedDictAction,
			default=get_default_kwargs(CTC),
			help="The keyword arguments for CTC class.",
		)
		group.add_argument(
			"--joint_net_conf",
			action=NestedDictAction,
			default=None,
			help="The keyword arguments for joint network class.",
		)

		group = parser.add_argument_group(description="Preprocess related")
		group.add_argument(
			"--use_preprocessor",
			type=str2bool,
			default=True,
			help="Apply preprocessing to data or not",
		)
		group.add_argument(
			"--token_type",
			type=str,
			default="bpe",
			choices=["bpe", "char", "word", "phn"],
			help="The text will be tokenized " "in the specified level token",
		)
		group.add_argument(
			"--bpemodel",
			type=str_or_none,
			default=None,
			help="The model file of sentencepiece",
		)
		parser.add_argument(
			"--non_linguistic_symbols",
			type=str_or_none,
			default=None,
			help="non_linguistic_symbols file path",
		)
		parser.add_argument(
			"--cleaner",
			type=str_or_none,
			choices=[None, "tacotron", "jaconv", "vietnamese"],
			default=None,
			help="Apply text cleaning",
		)
		parser.add_argument(
			"--g2p",
			type=str_or_none,
			choices=g2p_choices,
			default=None,
			help="Specify g2p method if --token_type=phn",
		)
		parser.add_argument(
			"--speech_volume_normalize",
			type=float_or_none,
			default=None,
			help="Scale the maximum amplitude to the given value.",
		)
		parser.add_argument(
			"--rir_scp",
			type=str_or_none,
			default=None,
			help="The file path of rir scp file.",
		)
		parser.add_argument(
			"--rir_apply_prob",
			type=float,
			default=1.0,
			help="THe probability for applying RIR convolution.",
		)
		parser.add_argument(
			"--noise_scp",
			type=str_or_none,
			default=None,
			help="The file path of noise scp file.",
		)
		parser.add_argument(
			"--noise_apply_prob",
			type=float,
			default=1.0,
			help="The probability applying Noise adding.",
		)
		parser.add_argument(
			"--noise_db_range",
			type=str,
			default="13_15",
			help="The range of noise decibel level.",
		)

		for class_choices in cls.class_choices_list:
			# Append --<name> and --<name>_conf.
			# e.g. --encoder and --encoder_conf
			class_choices.add_arguments(group)

	@classmethod
	def build_collate_fn(
		cls, args: argparse.Namespace, train: bool
	) -> Callable[
		[Collection[Tuple[str, Dict[str, np.ndarray]]]],
		Tuple[List[str], Dict[str, torch.Tensor]],
	]:
		assert check_argument_types()
		# NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
		return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

	@classmethod
	def build_preprocess_fn(
		cls, args: argparse.Namespace, train: bool
	) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
		assert check_argument_types()
		if args.use_preprocessor:
			retval = CommonPreprocessor(
				train=train,
				token_type=args.token_type,
				token_list=args.token_list,
				bpemodel=args.bpemodel,
				non_linguistic_symbols=args.non_linguistic_symbols,
				text_cleaner=args.cleaner,
				g2p_type=args.g2p,
				split_with_space=args.split_with_space if hasattr(args, "split_with_space") else False,
				# NOTE(kamo): Check attribute existence for backward compatibility
				rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
				rir_apply_prob=args.rir_apply_prob
				if hasattr(args, "rir_apply_prob")
				else 1.0,
				noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
				noise_apply_prob=args.noise_apply_prob
				if hasattr(args, "noise_apply_prob")
				else 1.0,
				noise_db_range=args.noise_db_range
				if hasattr(args, "noise_db_range")
				else "13_15",
				speech_volume_normalize=args.speech_volume_normalize
				if hasattr(args, "rir_scp")
				else None,
			)
		else:
			retval = None
		assert check_return_type(retval)
		return retval

	@classmethod
	def required_data_names(
		cls, train: bool = True, inference: bool = False
	) -> Tuple[str, ...]:
		if not inference:
			retval = ("speech", "text")
		else:
			# Recognition mode
			retval = ("speech",)
		return retval

	@classmethod
	def optional_data_names(
		cls, train: bool = True, inference: bool = False
	) -> Tuple[str, ...]:
		retval = ()
		assert check_return_type(retval)
		return retval

	@classmethod
	def build_model(cls, args: argparse.Namespace):
		assert check_argument_types()
		if isinstance(args.token_list, str):
			with open(args.token_list, encoding="utf-8") as f:
				token_list = [line.rstrip() for line in f]

			# Overwriting token_list to keep it as "portable".
			args.token_list = list(token_list)
		elif isinstance(args.token_list, (tuple, list)):
			token_list = list(args.token_list)
		else:
			raise RuntimeError("token_list must be str or list")
		vocab_size = len(token_list)
		logging.info(f"Vocabulary size: {vocab_size}")

		# 1. frontend
		if args.input_size is None:
			# Extract features in the model
			frontend_class = frontend_choices.get_class(args.frontend)
			frontend = frontend_class(**args.frontend_conf)
			input_size = frontend.output_size()
		else:
			# Give features from data-loader
			args.frontend = None
			args.frontend_conf = {}
			frontend = None
			input_size = args.input_size

		# 2. Data augmentation for spectrogram
		if args.specaug is not None:
			specaug_class = specaug_choices.get_class(args.specaug)
			specaug = specaug_class(**args.specaug_conf)
		else:
			specaug = None

		# 3. Normalization layer
		if args.normalize is not None:
			normalize_class = normalize_choices.get_class(args.normalize)
			normalize = normalize_class(**args.normalize_conf)
		else:
			normalize = None

		# 4. Pre-encoder input block
		# NOTE(kan-bayashi): Use getattr to keep the compatibility
		if getattr(args, "preencoder", None) is not None:
			preencoder_class = preencoder_choices.get_class(args.preencoder)
			preencoder = preencoder_class(**args.preencoder_conf)
			input_size = preencoder.output_size()
		else:
			preencoder = None

		# 5. Encoder
		encoder_class = encoder_choices.get_class(args.encoder)
		encoder = encoder_class(input_size=input_size, **args.encoder_conf)

		# 6. Post-encoder block
		# NOTE(kan-bayashi): Use getattr to keep the compatibility
		encoder_output_size = encoder.output_size()
		if getattr(args, "postencoder", None) is not None:
			postencoder_class = postencoder_choices.get_class(args.postencoder)
			postencoder = postencoder_class(
				input_size=encoder_output_size, **args.postencoder_conf
			)
			encoder_output_size = postencoder.output_size()
		else:
			postencoder = None

		# 7. Decoder
		decoder_class = decoder_choices.get_class(args.decoder)
		decoder = decoder_class(
			vocab_size=vocab_size,
			encoder_output_size=encoder_output_size,
			**args.decoder_conf,
		)

		# 8. CTC
		ctc = CTC(
			odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf
		)

		# 9. Build model
		try:
			model_class = model_choices.get_class(args.model)
		except AttributeError:
			model_class = model_choices.get_class("asr")
		model = model_class(
			vocab_size=vocab_size,
			frontend=frontend,
			specaug=specaug,
			normalize=normalize,
			preencoder=preencoder,
			encoder=encoder,
			postencoder=postencoder,
			decoder=decoder,
			ctc=ctc,
			token_list=token_list,
			**args.model_conf,
		)

		# 10. Initialize
		if args.init is not None:
			initialize(model, args.init)

		assert check_return_type(model)
		return model


class ASRTaskUniASR(ASRTask):
	# If you need more than one optimizers, change this value
	num_optimizers: int = 1

	# Add variable objects configurations
	class_choices_list = [
		# --frontend and --frontend_conf
		frontend_choices,
		# --specaug and --specaug_conf
		specaug_choices,
		# --normalize and --normalize_conf
		normalize_choices,
		# --model and --model_conf
		model_choices,
		# --preencoder and --preencoder_conf
		preencoder_choices,
		# --encoder and --encoder_conf
		encoder_choices,
		# --postencoder and --postencoder_conf
		postencoder_choices,
		# --decoder and --decoder_conf
		decoder_choices,
		# --predictor and --predictor_conf
		predictor_choices,
		# --encoder2 and --encoder2_conf
		encoder_choices2,
		# --decoder2 and --decoder2_conf
		decoder_choices2,
		# --predictor2 and --predictor2_conf
		predictor_choices2,
		# --stride_conv and --stride_conv_conf
		stride_conv_choices,
	]

	# If you need to modify train() or eval() procedures, change Trainer class here
	trainer = Trainer

	@classmethod
	def build_model(cls, args: argparse.Namespace):
		assert check_argument_types()
		if isinstance(args.token_list, str):
			with open(args.token_list, encoding="utf-8") as f:
				token_list = [line.rstrip() for line in f]

			# Overwriting token_list to keep it as "portable".
			args.token_list = list(token_list)
		elif isinstance(args.token_list, (tuple, list)):
			token_list = list(args.token_list)
		else:
			raise RuntimeError("token_list must be str or list")
		vocab_size = len(token_list)
		logging.info(f"Vocabulary size: {vocab_size}")

		# 1. frontend
		if args.input_size is None:
			# Extract features in the model
			frontend_class = frontend_choices.get_class(args.frontend)
			frontend = frontend_class(**args.frontend_conf)
			input_size = frontend.output_size()
		else:
			# Give features from data-loader
			args.frontend = None
			args.frontend_conf = {}
			frontend = None
			input_size = args.input_size

		# 2. Data augmentation for spectrogram
		if args.specaug is not None:
			specaug_class = specaug_choices.get_class(args.specaug)
			specaug = specaug_class(**args.specaug_conf)
		else:
			specaug = None

		# 3. Normalization layer
		if args.normalize is not None:
			normalize_class = normalize_choices.get_class(args.normalize)
			normalize = normalize_class(**args.normalize_conf)
		else:
			normalize = None

		# 4. Pre-encoder input block
		# NOTE(kan-bayashi): Use getattr to keep the compatibility
		if getattr(args, "preencoder", None) is not None:
			preencoder_class = preencoder_choices.get_class(args.preencoder)
			preencoder = preencoder_class(**args.preencoder_conf)
			input_size = preencoder.output_size()
		else:
			preencoder = None

		# 5. Encoder
		encoder_class = encoder_choices.get_class(args.encoder)
		encoder = encoder_class(input_size=input_size, **args.encoder_conf)
		encoder_output_size = encoder.output_size()

		stride_conv_class = stride_conv_choices.get_class(args.stride_conv)
		stride_conv = stride_conv_class(**args.stride_conv_conf, idim=input_size + encoder_output_size,
		                                odim=input_size + encoder_output_size)
		stride_conv_output_size = stride_conv.output_size()

		# 6. Encoder2
		encoder_class2 = encoder_choices2.get_class(args.encoder2)
		encoder2 = encoder_class2(input_size=stride_conv_output_size, **args.encoder2_conf)

		# 7. Post-encoder block
		# NOTE(kan-bayashi): Use getattr to keep the compatibility
		encoder_output_size2 = encoder2.output_size()
		if getattr(args, "postencoder", None) is not None:
			postencoder_class = postencoder_choices.get_class(args.postencoder)
			postencoder = postencoder_class(
				input_size=encoder_output_size, **args.postencoder_conf
			)
			encoder_output_size = postencoder.output_size()
		else:
			postencoder = None

		# 8. Decoder & Decoder2
		decoder_class = decoder_choices.get_class(args.decoder)
		decoder_class2 = decoder_choices2.get_class(args.decoder2)
		decoder = decoder_class(
			vocab_size=vocab_size,
			encoder_output_size=encoder_output_size,
			**args.decoder_conf,
		)
		decoder2 = decoder_class2(
			vocab_size=vocab_size,
			encoder_output_size=encoder_output_size2,
			**args.decoder2_conf,
		)

		# 9. CTC
		ctc = CTC(
			odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf
		)
		ctc2 = CTC(
			odim=vocab_size, encoder_output_size=encoder_output_size2, **args.ctc_conf
		)

		# 10. Predictor
		predictor_class = predictor_choices.get_class(args.predictor)
		predictor = predictor_class(**args.predictor_conf)

		predictor_class = predictor_choices2.get_class(args.predictor2)
		predictor2 = predictor_class(**args.predictor2_conf)

		# 11. Build model
		try:
			model_class = model_choices.get_class(args.model)
		except AttributeError:
			model_class = model_choices.get_class("asr")
		model = model_class(
			vocab_size=vocab_size,
			frontend=frontend,
			specaug=specaug,
			normalize=normalize,
			preencoder=preencoder,
			encoder=encoder,
			postencoder=postencoder,
			decoder=decoder,
			ctc=ctc,
			token_list=token_list,
			predictor=predictor,
			ctc2=ctc2,
			encoder2=encoder2,
			decoder2=decoder2,
			predictor2=predictor2,
			stride_conv=stride_conv,
			**args.model_conf,
		)

		# 12. Initialize
		if args.init is not None:
			initialize(model, args.init)

		assert check_return_type(model)
		return model


class ASRTaskParaformer(ASRTask):
	# If you need more than one optimizers, change this value
	num_optimizers: int = 1

	# Add variable objects configurations
	class_choices_list = [
		# --frontend and --frontend_conf
		frontend_choices,
		# --specaug and --specaug_conf
		specaug_choices,
		# --normalize and --normalize_conf
		normalize_choices,
		# --model and --model_conf
		model_choices,
		# --preencoder and --preencoder_conf
		preencoder_choices,
		# --encoder and --encoder_conf
		encoder_choices,
		# --postencoder and --postencoder_conf
		postencoder_choices,
		# --decoder and --decoder_conf
		decoder_choices,
		# --predictor and --predictor_conf
		predictor_choices,
	]

	# If you need to modify train() or eval() procedures, change Trainer class here
	trainer = Trainer

	@classmethod
	def build_model(cls, args: argparse.Namespace):
		assert check_argument_types()
		if isinstance(args.token_list, str):
			with open(args.token_list, encoding="utf-8") as f:
				token_list = [line.rstrip() for line in f]

			# Overwriting token_list to keep it as "portable".
			args.token_list = list(token_list)
		elif isinstance(args.token_list, (tuple, list)):
			token_list = list(args.token_list)
		else:
			raise RuntimeError("token_list must be str or list")
		vocab_size = len(token_list)
		logging.info(f"Vocabulary size: {vocab_size }")

		# 1. frontend
		if args.input_size is None:
			# Extract features in the model
			frontend_class = frontend_choices.get_class(args.frontend)
			frontend = frontend_class(**args.frontend_conf)
			input_size = frontend.output_size()
		else:
			# Give features from data-loader
			args.frontend = None
			args.frontend_conf = {}
			frontend = None
			input_size = args.input_size

		# 2. Data augmentation for spectrogram
		if args.specaug is not None:
			specaug_class = specaug_choices.get_class(args.specaug)
			specaug = specaug_class(**args.specaug_conf)
		else:
			specaug = None

		# 3. Normalization layer
		if args.normalize is not None:
			normalize_class = normalize_choices.get_class(args.normalize)
			normalize = normalize_class(**args.normalize_conf)
		else:
			normalize = None

		# 4. Pre-encoder input block
		# NOTE(kan-bayashi): Use getattr to keep the compatibility
		if getattr(args, "preencoder", None) is not None:
			preencoder_class = preencoder_choices.get_class(args.preencoder)
			preencoder = preencoder_class(**args.preencoder_conf)
			input_size = preencoder.output_size()
		else:
			preencoder = None

		# 5. Encoder
		encoder_class = encoder_choices.get_class(args.encoder)
		encoder = encoder_class(input_size=input_size, **args.encoder_conf)

		# 6. Post-encoder block
		# NOTE(kan-bayashi): Use getattr to keep the compatibility
		encoder_output_size = encoder.output_size()
		if getattr(args, "postencoder", None) is not None:
			postencoder_class = postencoder_choices.get_class(args.postencoder)
			postencoder = postencoder_class(
				input_size=encoder_output_size, **args.postencoder_conf
			)
			encoder_output_size = postencoder.output_size()
		else:
			postencoder = None

		# 7. Decoder
		decoder_class = decoder_choices.get_class(args.decoder)
		decoder = decoder_class(
			vocab_size=vocab_size,
			encoder_output_size=encoder_output_size,
			**args.decoder_conf,
		)

		# 8. CTC
		ctc = CTC(
			odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf
		)

		# 9. Predictor
		predictor_class = predictor_choices.get_class(args.predictor)
		predictor = predictor_class(**args.predictor_conf)

		# 10. Build model
		try:
			model_class = model_choices.get_class(args.model)
		except AttributeError:
			model_class = model_choices.get_class("asr")
		model = model_class(
			vocab_size=vocab_size,
			frontend=frontend,
			specaug=specaug,
			normalize=normalize,
			preencoder=preencoder,
			encoder=encoder,
			postencoder=postencoder,
			decoder=decoder,
			ctc=ctc,
			token_list=token_list,
			predictor=predictor,
			**args.model_conf,
		)

		# 11. Initialize
		if args.init is not None:
			initialize(model, args.init)

		assert check_return_type(model)
		return model
