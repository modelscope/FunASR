"""ASR Transducer Task."""

import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.frontend.default import DefaultFrontend
from funasr.models.frontend.windowing import SlidingWindow
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.models.specaug.specaug import SpecAug
from funasr.models.decoder.abs_decoder import AbsDecoder as AbsAttDecoder
from funasr.models.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder,
    TransformerDecoder,
)
from funasr.models.rnnt_predictor.abs_decoder import AbsDecoder
from funasr.models.rnnt_predictor.rnn_decoder import RNNDecoder
from funasr.models.rnnt_predictor.stateless_decoder import StatelessDecoder
from funasr.models.encoder.conformer_encoder import ConformerChunkEncoder
from funasr.models.e2e_transducer import TransducerModel
from funasr.models.e2e_transducer_unified import UnifiedTransducerModel
from funasr.models.joint_network import JointNetwork
from funasr.layers.abs_normalize import AbsNormalize
from funasr.layers.global_mvn import GlobalMVN
from funasr.layers.utterance_mvn import UtteranceMVN
from funasr.tasks.abs_task import AbsTask
from funasr.text.phoneme_tokenizer import g2p_choices
from funasr.train.class_choices import ClassChoices
from funasr.datasets.collate_fn import CommonCollateFn
from funasr.datasets.preprocessor import CommonPreprocessor
from funasr.train.trainer import Trainer
from funasr.utils.get_default_kwargs import get_default_kwargs
from funasr.utils.nested_dict_action import NestedDictAction
from funasr.utils.types import float_or_none, int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    "specaug",
    classes=dict(
        specaug=SpecAug,
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
    default="utterance_mvn",
    optional=True,
)
encoder_choices = ClassChoices(
        "encoder",
        classes=dict(
                chunk_conformer=ConformerChunkEncoder,
        ),
        default="chunk_conformer",
)

decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        rnn=RNNDecoder,
        stateless=StatelessDecoder,
    ),
    type_check=AbsDecoder,
    default="rnn",
)

att_decoder_choices = ClassChoices(
    "att_decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
    ),
    type_check=AbsAttDecoder,
    default=None,
    optional=True,
)
class ASRTransducerTask(AbsTask):
    """ASR Transducer Task definition."""

    num_optimizers: int = 1

    class_choices_list = [
        frontend_choices,
        specaug_choices,
        normalize_choices,
        encoder_choices,
        decoder_choices,
        att_decoder_choices,
    ]

    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """Add Transducer task arguments.
        Args:
            cls: ASRTransducerTask object.
            parser: Transducer arguments parser.
        """
        group = parser.add_argument_group(description="Task related.")

        # required = parser.get_default("required")
        # required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="Integer-string mapper for tokens.",
        )
        group.add_argument(
            "--split_with_space",
            type=str2bool,
            default=True,
            help="whether to split text using <space>",
        )
        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of dimensions for input features.",
        )
        group.add_argument(
            "--init",
            type=str_or_none,
            default=None,
            help="Type of model initialization to use.",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(TransducerModel),
            help="The keyword arguments for the model class.",
        )
        # group.add_argument(
        #     "--encoder_conf",
        #     action=NestedDictAction,
        #     default={},
        #     help="The keyword arguments for the encoder class.",
        # )
        group.add_argument(
            "--joint_network_conf",
            action=NestedDictAction,
            default={},
            help="The keyword arguments for the joint network class.",
        )
        group = parser.add_argument_group(description="Preprocess related.")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Whether to apply preprocessing to input data.",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The type of tokens to use during tokenization.",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The path of the sentencepiece model.",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="The 'non_linguistic_symbols' file path.",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Text cleaner to use.",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="g2p method to use if --token_type=phn.",
        )
        parser.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Normalization value for maximum amplitude scaling.",
        )
        parser.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The RIR SCP file path.",
        )
        parser.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="The probability of the applied RIR convolution.",
        )
        parser.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The path of noise SCP file.",
        )
        parser.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability of the applied noise addition.",
        )
        parser.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of the noise decibel level.",
        )
        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --decoder and --decoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        """Build collate function.
        Args:
            cls: ASRTransducerTask object.
            args: Task arguments.
            train: Training mode.
        Return:
            : Callable collate function.
        """
        assert check_argument_types()

        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """Build pre-processing function.
        Args:
            cls: ASRTransducerTask object.
            args: Task arguments.
            train: Training mode.
        Return:
            : Callable pre-processing function.
        """
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
        """Required data depending on task mode.
        Args:
            cls: ASRTransducerTask object.
            train: Training mode.
            inference: Inference mode.
        Return:
            retval: Required task data.
        """
        if not inference:
            retval = ("speech", "text")
        else:
            retval = ("speech",)

        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Optional data depending on task mode.
        Args:
            cls: ASRTransducerTask object.
            train: Training mode.
            inference: Inference mode.
        Return:
            retval: Optional task data.
        """
        retval = ()
        assert check_return_type(retval)

        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> TransducerModel:
        """Required data depending on task mode.
        Args:
            cls: ASRTransducerTask object.
            args: Task arguments.
        Return:
            model: ASR Transducer model.
        """
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

        # 4. Encoder
        
        if getattr(args, "encoder", None) is not None:
            encoder_class = encoder_choices.get_class(args.encoder)
            encoder = encoder_class(input_size, **args.encoder_conf)
        else:
            encoder = Encoder(input_size, **args.encoder_conf)
        encoder_output_size = encoder.output_size

        # 5. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)
        decoder = decoder_class(
            vocab_size,
            **args.decoder_conf,
        )
        decoder_output_size = decoder.output_size

        if getattr(args, "att_decoder", None) is not None:
            att_decoder_class = att_decoder_choices.get_class(args.att_decoder)

            att_decoder = att_decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=encoder_output_size,
                **args.att_decoder_conf,
            )
        else:
            att_decoder = None

        # 6. Joint Network
        joint_network = JointNetwork(
            vocab_size,
            encoder_output_size,
            decoder_output_size,
            **args.joint_network_conf,
        )

        # 7. Build model

        if encoder.unified_model_training:
            model = UnifiedTransducerModel(
                vocab_size=vocab_size,
                token_list=token_list,
                frontend=frontend,
                specaug=specaug,
                normalize=normalize,
                encoder=encoder,
                decoder=decoder,
                att_decoder=att_decoder,
                joint_network=joint_network,
                **args.model_conf,
            )

        else:
            model = TransducerModel(
                vocab_size=vocab_size,
                token_list=token_list,
                frontend=frontend,
                specaug=specaug,
                normalize=normalize,
                encoder=encoder,
                decoder=decoder,
                att_decoder=att_decoder,
                joint_network=joint_network,
                **args.model_conf,
            )

        # 8. Initialize model
        if args.init is not None:
            raise NotImplementedError(
                "Currently not supported.",
                "Initialization part will be reworked in a short future.",
            )

        #assert check_return_type(model)

        return model
