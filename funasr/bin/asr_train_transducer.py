#!/usr/bin/env python3

import os

from funasr.tasks.asr_transducer import ASRTransducerTask


# for ASR Training
def parse_args():
    parser = ASRTransducerTask.get_parser()
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="local gpu id.",
    )
    args = parser.parse_args()
    return args


def main(args=None, cmd=None):
    # for ASR Training
    ASRTransducerTask.main(args=args, cmd=cmd)


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
    if args.dataset_type == "small":
        if args.batch_size is not None:
            args.batch_size = args.batch_size * args.ngpu
        if args.batch_bins is not None:
            args.batch_bins = args.batch_bins * args.ngpu

    main(args=args)
