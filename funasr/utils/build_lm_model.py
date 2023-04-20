from funasr.lm.abs_model import AbsLM
from funasr.lm.seq_rnn_lm import SequentialRNNLM
from funasr.lm.transformer_lm import TransformerLM
from funasr.torch_utils.initialize import initialize
from funasr.train.class_choices import ClassChoices

lm_choices = ClassChoices(
    "lm",
    classes=dict(
        seq_rnn=SequentialRNNLM,
        transformer=TransformerLM,
    ),
    type_check=AbsLM,
    default="seq_rnn",
)

class_choices_list = [
    # --lm and --lm_conf
    lm_choices
]


def build_pretrain_model(args):
    # token_list
    if args.token_list is not None:
        with open(args.token_list) as f:
            token_list = [line.rstrip() for line in f]
        args.token_list = list(token_list)
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")
    else:
        vocab_size = None

    return model
