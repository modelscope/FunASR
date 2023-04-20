import argparse

from funasr.models.ctc import CTC
from funasr.utils.get_default_kwargs import get_default_kwargs
from funasr.utils.nested_dict_action import NestedDictAction
from funasr.utils.types import int_or_none
from funasr.utils.types import str2bool
from funasr.utils.types import str_or_none


def build_args(args):
    parser = argparse.ArgumentParser("Task related config")
    if args.task_name == "asr":
        from funasr.utils.build_asr_model import class_choices_list
        for class_choices in class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(parser)
        parser.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        parser.add_argument(
            "--split_with_space",
            type=str2bool,
            default=True,
            help="whether to split text using <space>",
        )
        parser.add_argument(
            "--seg_dict_file",
            type=str,
            default=None,
            help="seg_dict_file for text processing",
        )
        parser.add_argument(
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
        parser.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )
        parser.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )
        parser.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        parser.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--cmvn_file",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
    elif args.task_name == "pretrain":
        from funasr.utils.build_pretrain_model import class_choices_list
        for class_choices in class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(parser)
        parser.add_argument(
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
        parser.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )
        parser.add_argument(
            "--feats_type",
            type=str,
            default='fbank',
            help="feats type, e.g. fbank, wav, ark_wav(needed to be scale normalization)",
        )
        parser.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )
        parser.add_argument(
            "--pred_masked_weight",
            type=float,
            default=1.0,
            help="weight for predictive loss for masked frames",
        )
        parser.add_argument(
            "--pred_nomask_weight",
            type=float,
            default=0.0,
            help="weight for predictive loss for unmasked frames",
        )
        parser.add_argument(
            "--loss_weights",
            type=float,
            default=0.0,
            help="weights for additional loss terms (not first one)",
        )
    else:
        raise NotImplementedError("Not supported task: {}".format(args.task_name))

    args = parser.parse_args()
    return args
