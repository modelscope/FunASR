from __future__ import print_function
import numpy as np
import os
import kaldiio
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import math


class MultiProcessRunner:
    def __init__(self, fn):
        self.args = None
        self.process = fn

    def run(self):
        parser = argparse.ArgumentParser("")
        # Task-independent options
        parser.add_argument("--nj", type=int, default=16)
        parser.add_argument("--debug", action="store_true", default=False)
        parser.add_argument("--no_pbar", action="store_true", default=False)
        parser.add_argument("--verbose", action="store_ture", default=False)

        task_list, args = self.prepare(parser)
        result_list = self.pool_run(task_list, args)
        self.post(result_list, args)

    def prepare(self, parser):
        raise NotImplementedError("Please implement the prepare function.")

    def post(self, result_list, args):
        raise NotImplementedError("Please implement the post function.")

    def pool_run(self, tasks, args):
        results = []
        if args.debug:
            one_result = self.process(tasks[0])
            results.append(one_result)
        else:
            pool = Pool(args.nj)
            for one_result in tqdm(pool.imap(self.process, tasks), total=len(tasks), ascii=True, disable=args.no_pbar):
                results.append(one_result)
            pool.close()

        return results


class MultiProcessRunnerV2:
    def __init__(self, fn):
        self.args = None
        self.process = fn

    def run(self):
        parser = argparse.ArgumentParser("")
        # Task-independent options
        parser.add_argument("--nj", type=int, default=16)
        parser.add_argument("--debug", action="store_true", default=False)
        parser.add_argument("--no_pbar", action="store_true", default=False)
        parser.add_argument("--verbose", action="store_true", default=False)

        task_list, args = self.prepare(parser)
        chunk_size = int(math.ceil(float(len(task_list)) / args.nj))
        if args.verbose:
            print("Split {} tasks into {} sub-tasks with chunk_size {}".format(len(task_list), args.nj, chunk_size))
        subtask_list = [task_list[i*chunk_size: (i+1)*chunk_size] for i in range(args.nj)]
        result_list = self.pool_run(subtask_list, args)
        self.post(result_list, args)

    def prepare(self, parser):
        raise NotImplementedError("Please implement the prepare function.")

    def post(self, result_list, args):
        raise NotImplementedError("Please implement the post function.")

    def pool_run(self, tasks, args):
        results = []
        if args.debug:
            one_result = self.process(tasks[0])
            results.append(one_result)
        else:
            pool = Pool(args.nj)
            for one_result in tqdm(pool.imap(self.process, tasks), total=len(tasks), ascii=True, disable=args.no_pbar):
                results.append(one_result)
            pool.close()

        return results


class MultiProcessRunnerV3(MultiProcessRunnerV2):
    def run(self):
        parser = argparse.ArgumentParser("")
        # Task-independent options
        parser.add_argument("--nj", type=int, default=16)
        parser.add_argument("--debug", action="store_true", default=False)
        parser.add_argument("--no_pbar", action="store_true", default=False)
        parser.add_argument("--verbose", action="store_true", default=False)
        parser.add_argument("--sr", type=int, default=16000)

        task_list, shared_param, args = self.prepare(parser)
        chunk_size = int(math.ceil(float(len(task_list)) / args.nj))
        if args.verbose:
            print("Split {} tasks into {} sub-tasks with chunk_size {}".format(len(task_list), args.nj, chunk_size))
        subtask_list = [(i, task_list[i * chunk_size: (i + 1) * chunk_size], shared_param, args)
                        for i in range(args.nj)]
        result_list = self.pool_run(subtask_list, args)
        self.post(result_list, args)



class MyRunner(MultiProcessRunnerV3):
    def prepare(self, parser):
        assert isinstance(parser, argparse.ArgumentParser)
        parser.add_argument("enroll_dir", type=str)
        parser.add_argument("trial_in", type=str)
        parser.add_argument("trial_out", type=str)
        args = parser.parse_args()

        if not os.path.exists(os.path.dirname(args.trial_out)):
            os.makedirs(os.path.dirname(args.trial_out))

        flist_path = os.path.join(args.enroll_dir, "spk2xvec.flist")
        spk2xvec = {}
        for _path in open(flist_path, "r"):
            for key, value in kaldiio.load_ark(_path.strip()):
                if "-enroll" in key:
                    key = key.replace("-enroll", "")
                spk2xvec[key] = value

        flist_path = os.path.join(args.enroll_dir, "utt2xvec.flist")
        utt2xvec = {}
        for _path in open(flist_path, 'r'):
            for key, value in kaldiio.load_ark(_path.strip()):
                utt2xvec[key] = value

        task_list = [one_line.strip().split(" ") for one_line in open(args.trial_in, "rt")]
        return task_list, [spk2xvec, utt2xvec], args

    def post(self, results_list, args):
        with open(args.trial_out, "wt") as fs:
            for results in results_list:
                for one_item in results:
                    fs.write(one_item+"\n")


def process(task_args):
    task_id, task_list, [spk2xvec, utt2xvec], args = task_args
    results = []
    for spk, utt, _ in task_list:
        xvec = utt2xvec[utt]
        normed_x = xvec / np.linalg.norm(xvec)
        normed_y = spk2xvec[spk] / np.linalg.norm(spk2xvec[spk])
        score = np.sum(normed_x * normed_y)
        results.append("{} {} {:.5f}".format(spk, utt, score))

    return results


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
