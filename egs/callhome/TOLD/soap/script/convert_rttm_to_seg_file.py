import numpy as np
from funasr.utils.job_runner import MultiProcessRunnerV3
import os


class MyRunner(MultiProcessRunnerV3):

    def prepare(self, parser):
        parser.add_argument("--rttm_scp", type=str)
        parser.add_argument("--seg_file", type=str)
        args = parser.parse_args()

        if not os.path.exists(os.path.dirname(args.seg_file)):
            os.makedirs(os.path.dirname(args.seg_file))

        meeting2rttms = {}
        for one_line in open(args.rttm_scp, "rt"):
            parts = [x for x in one_line.strip().split(" ") if x != ""]
            mid, st, dur, spk_name = parts[1], float(parts[3]), float(parts[4]), parts[7]
            if mid not in meeting2rttms:
                meeting2rttms[mid] = []
            meeting2rttms[mid].append(one_line)

        task_list = list(meeting2rttms.items())
        return task_list, None, args

    def post(self, results_list, args):
        with open(args.seg_file, "wt") as fd:
            for results in results_list:
                fd.writelines(results)


def process(task_args):
    _, task_list, _, args = task_args
    outputs = []
    for mid, rttms in task_list:
        spk_turns = []
        length = 0
        for one_line in rttms:
            parts = one_line.strip().split(" ")
            _, st, dur, spk_name = parts[1], float(parts[3]), float(parts[4]), parts[7]
            st, ed = int(st*100), int((st + dur)*100)
            length = ed if ed > length else length
            spk_turns.append([mid, st, ed, spk_name])
        is_sph = np.zeros((length+1, ), dtype=bool)
        for _, st, ed, _ in spk_turns:
            is_sph[st:ed] = True

        st, in_speech = 0, False
        for i in range(length+1):
            if not in_speech and is_sph[i]:
                st, in_speech = i, True
            if in_speech and not is_sph[i]:
                in_speech = False
                outputs.append("{}-{:07d}-{:07d} {} {:.2f} {:.2f}\n".format(
                    mid, st, i, mid, float(st)/100, float(i)/100
                ))
    return outputs


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
