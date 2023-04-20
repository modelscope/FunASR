from funasr.utils.build_asr_model import build_asr_model


def build_model(args):
    if args.task_name == "asr":
        model = build_asr_model(args)
    else:
        raise NotImplementedError("Not supported task: {}".format(args.task_name))

    return model
