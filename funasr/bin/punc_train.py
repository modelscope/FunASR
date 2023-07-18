# -*- encoding: utf-8 -*-
#!/usr/bin/env python3
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

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

    if args.dataset_type == "small":
        if args.batch_size is not None:
            args.batch_size = args.batch_size * args.ngpu * args.num_worker_count
        if args.batch_bins is not None:
            args.batch_bins = args.batch_bins * args.ngpu * args.num_worker_count

    main(args=args)
