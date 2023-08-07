import os
import argparse
from funasr.utils.job_runner import MultiProcessRunnerV3
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict


class MyRunner(MultiProcessRunnerV3):
    def prepare(self, parser):
        assert isinstance(parser, argparse.ArgumentParser)
        parser.add_argument("wav_scp", type=str)
        parser.add_argument("out_dir", type=str)
        args = parser.parse_args()
        # assert args.sr == 8000, "For callhome dataset, the sample rate should be 8000, use --sr 8000."

        wav_scp = load_scp_as_list(args.wav_scp)
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        return wav_scp, None, args

    def post(self, result_list, args):
        count = [0, 0]
        for result in result_list:
            count[0] += result[0]
            count[1] += result[1]
        print("All threads done, {} success, {} failed.".format(count[0], count[1]))


def process(task_args):
    task_id, task_list, _, args = task_args

    count = [0, 0]
    for utt_id, cmd in task_list:
        try:
            wav_path = os.path.join(args.out_dir, "{}.wav".format(utt_id))
            cmd = cmd.replace("|", "> {}".format(wav_path))
            os.system(cmd)
            count[0] += 1
        except:
            print("Failed execute command for {}.".format(utt_id))
            count[1] += 1

    return count


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
