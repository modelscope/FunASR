import argparse
import json
import os

import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(
        description="combine cmvn file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dim",
        default=80,
        type=int,
        help="feature dim",
    )
    parser.add_argument(
        "--cmvn_dir",
        default=False,
        required=True,
        type=str,
        help="cmvn dir",
    )

    parser.add_argument(
        "--nj",
        default=1,
        required=True,
        type=int,
        help="num of cmvn files",
    )
    parser.add_argument(
        "--output_dir",
        default=False,
        required=True,
        type=str,
        help="output dir",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    total_means = np.zeros(args.dim)
    total_vars = np.zeros(args.dim)
    total_frames = 0

    cmvn_file = os.path.join(args.output_dir, "cmvn.json")

    for i in range(1, args.nj + 1):
        with open(os.path.join(args.cmvn_dir, "cmvn.{}.json".format(str(i)))) as fin:
            cmvn_stats = json.load(fin)

        total_means += np.array(cmvn_stats["mean_stats"])
        total_vars += np.array(cmvn_stats["var_stats"])
        total_frames += cmvn_stats["total_frames"]

    cmvn_info = {
        'mean_stats': list(total_means.tolist()),
        'var_stats': list(total_vars.tolist()),
        'total_frames': total_frames
    }
    with open(cmvn_file, 'w') as fout:
        fout.write(json.dumps(cmvn_info))


if __name__ == '__main__':
    main()
