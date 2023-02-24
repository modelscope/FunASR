import logging
import numpy as np
import soundfile
import kaldiio
from funasr.utils.job_runner import MultiProcessRunnerV3
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
import os
import argparse
from collections import OrderedDict


class MyRunner(MultiProcessRunnerV3):

    def prepare(self, parser: argparse.ArgumentParser):
        parser.add_argument("--input_scp", type=str, required=True)
        parser.add_argument("--out_path")
        args = parser.parse_args()

        if not os.path.exists(os.path.dirname(args.out_path)):
            os.makedirs(os.path.dirname(args.out_path))

        task_list = load_scp_as_list(args.input_scp)
        return task_list, None, args

    def post(self, result_list, args):
        fd = open(args.out_path, "wt", encoding="utf-8")
        for results in result_list:
            for uttid, shape in results:
                fd.write("{} {}\n".format(uttid, ",".join(shape)))
        fd.close()


def process(task_args):
    task_idx, task_list, _, args = task_args
    rst = []
    for uttid, file_path in task_list:
        data = kaldiio.load_mat(file_path)
        shape = [str(x) for x in data.shape]
        rst.append((uttid, shape))
    return rst


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
