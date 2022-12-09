from kaldiio import ReadHelper

import argparse
import numpy as np
import json


def get_parser():
    parser = argparse.ArgumentParser(
        description="computer global cmvn",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dims",
        "-d",
        default=80,
        type=int,
        help="feature dims",
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

    ark_file = args.ark_file + "/feats." + str(args.ark_index) + ".ark"
    cmvn_file = args.output_dir + "/cmvn." + str(args.ark_index) + ".json"

    mean_stats = np.zeros(args.dims)
    var_stats = np.zeros(args.dims)
    total_frames = 0

    with ReadHelper('ark:{}'.format(ark_file)) as ark_reader:
        for key, mat in ark_reader:
            mean_stats += np.sum(mat, axis=0)
            var_stats += np.sum(np.square(mat), axis=0)
            total_frames += mat.shape[0]

    cmvn_info = {
        'mean_stats': list(mean_stats.tolist()),
        'var_stats': list(var_stats.tolist()),
        'total_frames': total_frames
    }
    with open(cmvn_file, 'w') as fout:
        fout.write(json.dumps(cmvn_info))


if __name__ == '__main__':
    main()
