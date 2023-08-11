from funasr.build_utils.build_asr_model import build_asr_model
from funasr.build_utils.build_diar_model import build_diar_model
from funasr.build_utils.build_lm_model import build_lm_model
from funasr.build_utils.build_pretrain_model import build_pretrain_model
from funasr.build_utils.build_punc_model import build_punc_model
from funasr.build_utils.build_sv_model import build_sv_model
from funasr.build_utils.build_vad_model import build_vad_model
from funasr.build_utils.build_ss_model import build_ss_model


def build_model(args):
    if args.task_name == "asr":
        model = build_asr_model(args)
    elif args.task_name == "pretrain":
        model = build_pretrain_model(args)
    elif args.task_name == "lm":
        model = build_lm_model(args)
    elif args.task_name == "punc":
        model = build_punc_model(args)
    elif args.task_name == "vad":
        model = build_vad_model(args)
    elif args.task_name == "diar":
        model = build_diar_model(args)
    elif args.task_name == "sv":
        model = build_sv_model(args)
    elif args.task_name == "ss":
        model = build_ss_model(args)
    else:
        raise NotImplementedError("Not supported task: {}".format(args.task_name))

    return model
