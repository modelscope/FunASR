import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--split_num", type=int, default=64)
    args = parser.parse_args()
    root_path = args.root_path
    out_path = args.out_path
    datasets = ["train", "dev"]
    split_num = args.split_num

    for dataset in datasets:
        with open(os.path.join(out_path, dataset, "feats.scp"), "w") as out_f:
            for i in range(split_num):
                idx = str(i + 1)
                feature_file = os.path.join(root_path, dataset, "feature.scp.{}".format(idx))
                label_file = os.path.join(root_path, dataset, "label.scp.{}".format(idx))
                with open(feature_file) as ff, open(label_file) as fl:
                    ff_lines = ff.readlines()
                    fl_lines = fl.readlines()
                    for ff_line, fl_line in zip(ff_lines, fl_lines):
                        sample_name, f_path = ff_line.strip().split()
                        _, l_path = fl_line.strip().split()
                        out_f.write("{} {} {}\n".format(sample_name, f_path, l_path))