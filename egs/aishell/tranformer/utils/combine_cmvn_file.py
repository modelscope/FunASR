import argparse
import json
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(
        description="combine cmvn file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cmvn-dir",
        "-c",
        default=False,
        required=True,
        type=str,
        help="cmvn dir",
    )

    parser.add_argument(
        "--nj",
        "-n",
        default=1,
        required=True,
        type=int,
        help="num of cmvn file",
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

    total_means = 0.0
    total_vars = 0.0
    total_frames = 0

    cmvn_file = args.output_dir + "/cmvn.json"

    for i in range(1, args.nj+1):
        with open(args.cmvn_dir + "/cmvn." + str(i) + ".json", "r") as fin:
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
