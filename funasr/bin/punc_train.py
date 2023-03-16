#!/usr/bin/env python3
import os
from funasr.tasks.punctuation import PunctuationTask


def parse_args():
    parser = PunctuationTask.get_parser()
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="local gpu id.",
    )
    parser.add_argument(
        "--punc_list",
        type=str,
        default=None,
        help="Punctuation list",
    )
    args = parser.parse_args()
    return args


def main(args=None, cmd=None):
    """
    punc training.
    """
    PunctuationTask.main(args=args, cmd=cmd)


if __name__ == "__main__":
    args = parse_args()

    # setup local gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # DDP settings
    if args.ngpu > 1:
        args.distributed = True
    else:
        args.distributed = False

    main(args=args)
