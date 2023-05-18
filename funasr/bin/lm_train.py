# -*- encoding: utf-8 -*-
#!/usr/bin/env python3
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os

from funasr.tasks.lm import LMTask


# for LM Training
def parse_args():
    parser = LMTask.get_parser()
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="local gpu id.",
    )
    args = parser.parse_args()
    return args


def main(args=None, cmd=None):
    # for LM Training
    LMTask.main(args=args, cmd=cmd)


if __name__ == '__main__':
    args = parse_args()

    # setup local gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # DDP settings
    if args.ngpu > 1:
        args.distributed = True
    else:
        args.distributed = False
    assert args.num_worker_count == 1

    # re-compute batch size: when dataset type is small
    if args.dataset_type == "small" and args.ngpu != 0:
        if args.batch_size is not None:
            args.batch_size = args.batch_size * args.ngpu
        if args.batch_bins is not None:
            args.batch_bins = args.batch_bins * args.ngpu

    main(args=args)
