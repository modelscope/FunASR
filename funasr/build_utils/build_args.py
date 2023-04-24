import argparse

from funasr.models.ctc import CTC
from funasr.utils.get_default_kwargs import get_default_kwargs
from funasr.utils.nested_dict_action import NestedDictAction
from funasr.utils.types import int_or_none
from funasr.utils.types import str2bool
from funasr.utils.types import str_or_none


def build_args(args, extra_task_params):
    parser = argparse.ArgumentParser("Task related config")
    if args.task_name == "asr":
        from funasr.build_utils.build_asr_model import class_choices_list
        for class_choices in class_choices_list:
            class_choices.add_arguments(parser)
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
            "--cmvn_file",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )

    elif args.task_name == "pretrain":
        from funasr.build_utils.build_pretrain_model import class_choices_list
        for class_choices in class_choices_list:
            class_choices.add_arguments(parser)
        parser.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

    elif args.task_name == "lm":
        from funasr.build_utils.build_lm_model import class_choices_list
        for class_choices in class_choices_list:
            class_choices.add_arguments(parser)

    elif args.task_name == "punc":
        from funasr.build_utils.build_punc_model import class_choices_list
        for class_choices in class_choices_list:
            class_choices.add_arguments(parser)

    elif args.task_name == "vad":
        from funasr.build_utils.build_vad_model import class_choices_list
        for class_choices in class_choices_list:
            class_choices.add_arguments(parser)
        parser.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

    elif args.task_name == "diar":
        from funasr.build_utils.build_diar_model import class_choices_list
        for class_choices in class_choices_list:
            class_choices.add_arguments(parser)

    else:
        raise NotImplementedError("Not supported task: {}".format(args.task_name))

    task_args = parser.parse_args(extra_task_params)
    return task_args
