import argparse
import json
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(
        description="cmvn converter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cmvn_json",
        default=False,
        required=True,
        type=str,
        help="cmvn json file",
    )
    parser.add_argument(
        "--am_mvn",
        default=False,
        required=True,
        type=str,
        help="am mvn file",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(args.cmvn_json, "r") as fin:
        cmvn_dict = json.load(fin)

    mean_stats = np.array(cmvn_dict["mean_stats"])
    var_stats = np.array(cmvn_dict["var_stats"])
    total_frame = np.array(cmvn_dict["total_frames"])

    mean = -1.0 * mean_stats / total_frame
    var = 1.0 / np.sqrt(var_stats / total_frame - mean * mean)
    dims = mean.shape[0]
    with open(args.am_mvn, 'w') as fout:
        fout.write("<Nnet>" + "\n" + "<Splice> " + str(dims) + " " + str(dims) + '\n' + "[ 0 ]" + "\n" + "<AddShift> " + str(dims) + " " + str(dims) + "\n")
        mean_str = str(list(mean)).replace(',', '').replace('[', '[ ').replace(']', ' ]')
        fout.write("<LearnRateCoef> 0 " + mean_str + '\n')
        fout.write("<Rescale> " + str(dims) + " " + str(dims) + '\n')
        var_str = str(list(var)).replace(',', '').replace('[', '[ ').replace(']', ' ]')
        fout.write("<LearnRateCoef> 0 " + var_str + '\n')
        fout.write("</Nnet>" + '\n')

if __name__ == '__main__':
    main()
