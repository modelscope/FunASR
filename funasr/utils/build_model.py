from funasr.utils.build_asr_model import build_asr_model
from funasr.utils.build_pretrain_model import build_pretrain_model


def build_model(args):
    if args.task_name == "asr":
        model = build_asr_model(args)
    elif args.task_name == "pretrain":
        model = build_pretrain_model(args)
    elif args.task_name == "lm":
        model = build_lm_model(args)
    else:
        raise NotImplementedError("Not supported task: {}".format(args.task_name))

    return model
