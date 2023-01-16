from kaldiio import ReadHelper
from kaldiio import WriteHelper

import argparse
import json
import math
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(
        description="apply cmvn",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ark-file",
        "-a",
        default=False,
        required=True,
        type=str,
        help="fbank ark file",
    )
    parser.add_argument(
        "--cmvn-file",
        "-c",
        default=False,
        required=True,
        type=str,
        help="cmvn file",
    )
    parser.add_argument(
        "--ark-index",
        "-i",
        default=1,
        required=True,
        type=int,
        help="ark index",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=False,
        required=True,
        type=str,
        help="output dir",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    ark_file = args.output_dir + "/feats." + str(args.ark_index) + ".ark"
    scp_file = args.output_dir + "/feats." + str(args.ark_index) + ".scp"
    ark_writer = WriteHelper('ark,scp:{},{}'.format(ark_file, scp_file))

    with open(args.cmvn_file) as f:
        cmvn_stats = json.load(f)

    means = cmvn_stats['mean_stats']
    vars = cmvn_stats['var_stats']
    total_frames = cmvn_stats['total_frames']

    for i in range(len(means)):
        means[i] /= total_frames
        vars[i] = vars[i] / total_frames - means[i] * means[i]
        if vars[i] < 1.0e-20:
            vars[i] = 1.0e-20
        vars[i] = 1.0 / math.sqrt(vars[i])

    with ReadHelper('ark:{}'.format(args.ark_file)) as ark_reader:
        for key, mat in ark_reader:
            mat = (mat - means) * vars
            ark_writer(key, mat)


if __name__ == '__main__':
    main()
