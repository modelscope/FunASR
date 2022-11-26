#!/usr/bin/env python3
from funasr.tasks.lm import LMTask


def get_parser():
    parser = LMTask.get_parser()
    return parser


def main(cmd=None):
    """LM training.

    Example:

        % python lm_train.py asr --print_config --optim adadelta
        % python lm_train.py --config conf/train_asr.yaml
    """
    LMTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
