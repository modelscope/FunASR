import logging

from funasr.models.target_delay_transformer import TargetDelayTransformer
from funasr.models.vad_realtime_transformer import VadRealtimeTransformer
from funasr.torch_utils.initialize import initialize
from funasr.train.abs_model import PunctuationModel
from funasr.train.class_choices import ClassChoices

punc_choices = ClassChoices(
    "punctuation",
    classes=dict(
        target_delay=TargetDelayTransformer,
        vad_realtime=VadRealtimeTransformer
    ),
    default="target_delay",
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        punc=PunctuationModel,
    ),
    default="punc",
)
class_choices_list = [
    # --punc and --punc_conf
    punc_choices,
    # --model and --model_conf
    model_choices
]


def build_punc_model(args):
    # token_list and punc list
    if isinstance(args.token_list, str):
        with open(args.token_list, encoding="utf-8") as f:
            token_list = [line.rstrip() for line in f]
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

    # punc
    punc_class = punc_choices.get_class(args.punctuation)
    punc = punc_class(vocab_size=vocab_size, punc_size=punc_size, **args.punctuation_conf)

    if "punc_weight" in args.model_conf:
        args.model_conf.pop("punc_weight")
    model = PunctuationModel(punc_model=punc, vocab_size=vocab_size, punc_weight=punc_weight_list, **args.model_conf)

    # initialize
    if args.init is not None:
        initialize(model, args.init)

    return model
