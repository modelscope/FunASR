from __future__ import print_function
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
