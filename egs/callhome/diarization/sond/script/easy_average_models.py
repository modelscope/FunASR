import os
import sys
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        default=None,
        type=str,
        help="Director contains saved models."
    )
    parser.add_argument(
        "--average_epochs",
        nargs="+",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default="der",
        help="The metric name of best models, only used for name."
    )
    args = parser.parse_args()

    root_path = args.model_dir
    idx_list = args.average_epochs
    n_models = len(idx_list)
    metric = args.metric_name

    if n_models > 0:
        avg = None
        for idx in idx_list:
            model_file = os.path.join(root_path, "{}epoch.pth".format(str(idx)))
            states = torch.load(model_file, map_location="cpu")
            if avg is None:
                avg = states
            else:
                for k in avg:
                    avg[k] = avg[k] + states[k]

        for k in avg:
            if str(avg[k].dtype).startswith("torch.int"):
                pass
            else:
                avg[k] = avg[k] / n_models

        output_file = os.path.join(root_path, "valid.{}.ave_{}best.pth".format(metric, n_models))
        torch.save(avg, output_file)
    else:
        print("Number of models to average is 0, skip.")
