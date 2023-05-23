from funasr.models.ctc import CTC
from funasr.utils import config_argparse
from funasr.utils.get_default_kwargs import get_default_kwargs
from funasr.utils.nested_dict_action import NestedDictAction
from funasr.utils.types import int_or_none
from funasr.utils.types import str2bool
from funasr.utils.types import str_or_none


def build_args(args, parser, extra_task_params):
    task_parser = config_argparse.ArgumentParser("Task related config")
    if args.task_name == "asr":
        from funasr.build_utils.build_asr_model import class_choices_list
        for class_choices in class_choices_list:
            class_choices.add_arguments(task_parser)
        task_parser.add_argument(
            "--split_with_space",
            type=str2bool,
            default=True,
            help="whether to split text using <space>",
        )
        task_parser.add_argument(
            "--seg_dict_file",
            type=str,
            default=None,
            help="seg_dict_file for text processing",
        )
        task_parser.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )
        task_parser.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )
        task_parser.add_argument(
            "--cmvn_file",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )

    elif args.task_name == "pretrain":
        from funasr.build_utils.build_pretrain_model import class_choices_list
        for class_choices in class_choices_list:
            class_choices.add_arguments(task_parser)
        task_parser.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

    elif args.task_name == "lm":
        from funasr.build_utils.build_lm_model import class_choices_list
        for class_choices in class_choices_list:
            class_choices.add_arguments(task_parser)

    elif args.task_name == "punc":
        from funasr.build_utils.build_punc_model import class_choices_list
        for class_choices in class_choices_list:
            class_choices.add_arguments(task_parser)

    elif args.task_name == "vad":
        from funasr.build_utils.build_vad_model import class_choices_list
        for class_choices in class_choices_list:
            class_choices.add_arguments(task_parser)
        task_parser.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

    elif args.task_name == "diar":
        from funasr.build_utils.build_diar_model import class_choices_list
        for class_choices in class_choices_list:
            class_choices.add_arguments(task_parser)

    else:
        raise NotImplementedError("Not supported task: {}".format(args.task_name))

    for action in parser._actions:
        if not any(action.dest == a.dest for a in task_parser._actions):
            task_parser._add_action(action)

    task_parser.set_defaults(**vars(args))
    task_args = task_parser.parse_args(extra_task_params)
    return task_args
