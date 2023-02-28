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
from funasr.datasets.preprocessor import PuncTrainTokenizerCommonPreprocessor
from funasr.punctuation.abs_model import AbsPunctuation
from funasr.punctuation.espnet_model import ESPnetPunctuationModel
from funasr.punctuation.target_delay_transformer import TargetDelayTransformer
from funasr.punctuation.vad_realtime_transformer import VadRealtimeTransformer
from funasr.tasks.abs_task import AbsTask
from funasr.text.phoneme_tokenizer import g2p_choices
from funasr.torch_utils.initialize import initialize
from funasr.train.class_choices import ClassChoices
from funasr.train.trainer import Trainer
from funasr.utils.get_default_kwargs import get_default_kwargs
from funasr.utils.nested_dict_action import NestedDictAction
from funasr.utils.types import str2bool
from funasr.utils.types import str_or_none

punc_choices = ClassChoices(
    "punctuation",
    classes=dict(target_delay=TargetDelayTransformer, vad_realtime=VadRealtimeTransformer),
    type_check=AbsPunctuation,
    default="target_delay",
)


class PunctuationTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [punc_choices]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        assert check_argument_types()
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
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
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetPunctuationModel),
            help="The keyword arguments for model class.",
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
            choices=["bpe", "char", "word"],
            help="",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file fo sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
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

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

        assert check_return_type(parser)
        return parser

    @classmethod
    def build_collate_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn(int_pad_value=0)

    @classmethod
    def build_preprocess_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        token_types = [args.token_type, args.token_type]
        token_lists = [args.token_list, args.punc_list]
        bpemodels = [args.bpemodel, args.bpemodel]
        text_names = ["text", "punc"]
        if args.use_preprocessor:
            retval = PuncTrainTokenizerCommonPreprocessor(
                train=train,
                token_type=token_types,
                token_list=token_lists,
                bpemodel=bpemodels,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                text_name = text_names,
                non_linguistic_symbols=args.non_linguistic_symbols,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("text", "punc")
        if inference:
            retval = ("text", )
        return retval

    @classmethod
    def optional_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("vad",)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetPunctuationModel:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # "args" is saved as it is in a yaml file by BaseTask.main().
            # Overwriting token_list to keep it as "portable".
            args.token_list = token_list.copy()
        if isinstance(args.punc_list, str):
            with open(args.punc_list, encoding="utf-8") as f2:
                pairs = [line.rstrip().split(":") for line in f2]
            punc_list = [pair[0] for pair in pairs]
            punc_weight_list = [float(pair[1]) for pair in pairs]
            args.punc_list = punc_list.copy()
        elif isinstance(args.punc_list, list):
            punc_list = args.punc_list.copy()
            punc_weight_list = [1] * len(punc_list)
        if isinstance(args.token_list, (tuple, list)):
            token_list = args.token_list.copy()
        else:
            raise RuntimeError("token_list must be str or dict")

        vocab_size = len(token_list)
        punc_size = len(punc_list)
        logging.info(f"Vocabulary size: {vocab_size}")

        # 1. Build PUNC model
        punc_class = punc_choices.get_class(args.punctuation)
        punc = punc_class(vocab_size=vocab_size, punc_size=punc_size, **args.punctuation_conf)

        # 2. Build ESPnetModel
        # Assume the last-id is sos_and_eos
        if "punc_weight" in args.model_conf:
            args.model_conf.pop("punc_weight")
        model = ESPnetPunctuationModel(punc_model=punc, vocab_size=vocab_size, punc_weight=punc_weight_list, **args.model_conf)

        # FIXME(kamo): Should be done in model?
        # 3. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
