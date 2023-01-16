from kaldiio import ReadHelper, WriteHelper

import argparse
import numpy as np


def build_LFR_features(inputs, m=7, n=6):
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    left_padding = np.tile(inputs[0], ((m - 1) // 2, 1))
    inputs = np.vstack((left_padding, inputs))
    T = T + (m - 1) // 2
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
        else:
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i * n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)


def build_CMVN_features(inputs, mvn_file):  # noqa
    with open(mvn_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    add_shift_list = []
    rescale_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == '<AddShift>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                add_shift_line = line_item[3:(len(line_item) - 1)]
                add_shift_list = list(add_shift_line)
                continue
        elif line_item[0] == '<Rescale>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                rescale_line = line_item[3:(len(line_item) - 1)]
                rescale_list = list(rescale_line)
                continue

    for j in range(inputs.shape[0]):
        for k in range(inputs.shape[1]):
            add_shift_value = add_shift_list[k]
            rescale_value = rescale_list[k]
            inputs[j, k] = float(inputs[j, k]) + float(add_shift_value)
            inputs[j, k] = float(inputs[j, k]) * float(rescale_value)

    return inputs


def get_parser():
    parser = argparse.ArgumentParser(
        description="apply low_frame_rate and cmvn",
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
        "--lfr",
        "-f",
        default=True,
        type=str,
        help="low frame rate",
    )
    parser.add_argument(
        "--lfr-m",
        "-m",
        default=7,
        type=int,
        help="number of frames to stack",
    )
    parser.add_argument(
        "--lfr-n",
        "-n",
        default=6,
        type=int,
        help="number of frames to skip",
    )
    parser.add_argument(
        "--cmvn-file",
        "-c",
        default=False,
        required=True,
        type=str,
        help="global cmvn file",
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

    dump_ark_file = args.output_dir + "/feats." + str(args.ark_index) + ".ark"
    dump_scp_file = args.output_dir + "/feats." + str(args.ark_index) + ".scp"
    shape_file = args.output_dir + "/len." + str(args.ark_index)
    ark_writer = WriteHelper('ark,scp:{},{}'.format(dump_ark_file, dump_scp_file))

    shape_writer = open(shape_file, 'w')
    with ReadHelper('ark:{}'.format(args.ark_file)) as ark_reader:
        for key, mat in ark_reader:
            if args.lfr:
                lfr = build_LFR_features(mat, args.lfr_m, args.lfr_n)
            else:
                lfr = mat
            cmvn = build_CMVN_features(lfr, args.cmvn_file)
            dims = cmvn.shape[1]
            lens = cmvn.shape[0]
            shape_writer.write(key + " " + str(lens) + "," + str(dims) + '\n')
            ark_writer(key, cmvn)


if __name__ == '__main__':
    main()

